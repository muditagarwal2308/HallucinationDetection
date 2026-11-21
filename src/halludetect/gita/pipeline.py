"""End-to-end retrieval-augmented QA on the Bhagavad Gita with verse-level verification."""

from __future__ import annotations

import time

from ..llm import prompts
from ..llm.base import LLM
from .corpus import GitaCorpus
from .policy import RepairPolicy
from .simulator import SimulatedGitaLLM
from .verifier import VerseVerifier, tuned_support_threshold


class GitaQA:
    """Retrieve verses → generate a cited answer → verify verse by verse → accept, repair, regenerate or abstain.

    ``llm`` may be any :class:`~halludetect.llm.base.LLM` (e.g. an Ollama model through
    :class:`~halludetect.llm.openai_compat.OpenAICompatLLM`) or ``None`` for the offline
    simulator, whose ``mode`` injects a chosen hallucination type for demonstration.
    """

    def __init__(self, corpus: GitaCorpus | None = None, llm: LLM | None = None,
                 verifier: VerseVerifier | None = None, k: int = 5, simulator_mode: str = "none"):
        self.corpus = corpus or GitaCorpus.load()
        self.llm = llm
        self.simulator = SimulatedGitaLLM(self.corpus, mode=simulator_mode)
        self.verifier = verifier or VerseVerifier(self.corpus, support_threshold=tuned_support_threshold("lexical"))
        self.policy = RepairPolicy(self.corpus)
        self.k = k

    def _generate(self, question: str, verses, feedback: str | None = None) -> tuple[str, str]:
        if self.llm is None:
            return self.simulator.compose(question, verses)
        prompt = prompts.gita_prompt(question, [(v.ref, v.english) for v in verses])
        if feedback:
            prompt += f"\n\nA verifier found problems in your previous answer: {feedback}\nRewrite the answer so " \
                      "that every claim is supported by the verse it cites, or say that the verses do not answer it."
        text = self.llm.generate(prompt, n=1, temperature=0.0, max_tokens=220, system=prompts.GITA_SYSTEM)[0].text
        return text.strip(), "llm"

    def ask(self, question: str, simulator_mode: str | None = None) -> dict:
        t0 = time.time()
        hits = self.corpus.search(question, k=self.k)
        verses = [v for v, _ in hits]
        if simulator_mode is not None:
            self.simulator.mode = simulator_mode
        answer, injected = self._generate(question, verses)
        report = self.verifier.verify(question, answer)
        decision = self.policy.decide(report)
        final_report = report
        regenerated = False
        if self.llm is not None and report.hallucinated and decision.action in ("repair", "abstain") \
                and not report.premise.is_problematic:
            feedback = "; ".join(i.message for i in report.issues)
            answer2, _ = self._generate(question, verses, feedback)
            rep2 = self.verifier.verify(question, answer2)
            if not rep2.hallucinated:
                regenerated, final_report = True, rep2
                decision = self.policy.decide(rep2)
                decision.action = "regenerate"
                decision.rationale = "The LLM rewrote the answer after verifier feedback."
        elif decision.action == "repair":
            final_report = self.verifier.verify(question, decision.answer)
        return {
            "question": question,
            "retrieved": [{"ref": v.ref, "score": round(s, 3), "speaker": v.speaker, "english": v.english,
                           "iast": v.iast} for v, s in hits],
            "answer": answer, "injected": injected, "report": report.as_dict(), "decision": decision.as_dict(),
            "final_report": final_report.as_dict(), "regenerated": regenerated,
            "seconds": round(time.time() - t0, 3),
        }
