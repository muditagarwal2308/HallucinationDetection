"""Narrated, fully offline terminal demo of UG-RAV and VGHD (companion to docs/DEMO_SCRIPT.md).

Usage::

    python scripts/run_demo.py              # all scenes, no pauses (about 10 s)
    python scripts/run_demo.py --pause      # wait for Enter between scenes (live presentation)
    python scripts/run_demo.py --scene 3    # a single scene
    python scripts/run_demo.py --no-color   # plain text (logs, CI)

Scenes
------
1. The problem: a fluent wrong answer and semantic entropy over sampled answers
2. UG-RAV corrects a high-uncertainty error with verified evidence (Godan)
3. UG-RAV catches a *confident* error that semantic entropy misses (Anandamath)
4. UG-RAV accepts a well-known fact without retrieval, and abstains when nothing helps
5. VGHD verifies a cited Gita answer and repairs the wrong verse number (BG 2.20 -> 2.23)
6. VGHD catalogue: wrong speaker, misquotation, false premise, out-of-scope
7. Gita RAG assistant: an injected fabricated citation is detected and repaired
8. Headline results from the benchmark files

Everything runs on the offline simulator and the lexical NLI, so the demo needs no GPU and no network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halludetect.gita import GitaCorpus, VerseVerifier  # noqa: E402
from halludetect.gita.pipeline import GitaQA  # noqa: E402
from halludetect.gita.policy import RepairPolicy  # noqa: E402
from halludetect.gita.verifier import tuned_support_threshold  # noqa: E402
from halludetect.nli import LexicalNLI  # noqa: E402
from halludetect.ugrav import UGRAV, load_passages, load_world  # noqa: E402
from halludetect.ugrav.data import PassageRetriever  # noqa: E402
from halludetect.ugrav.simulator import SimulatedQALLM  # noqa: E402
from halludetect.uncertainty import cluster_answers, lexical_entropy, semantic_entropy  # noqa: E402

COLOR = True
WIDTH = 100


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def amber(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def say(text: str, indent: int = 2) -> None:
    pad = " " * indent
    print(textwrap.fill(text, WIDTH, initial_indent=pad, subsequent_indent=pad))


def kv(key: str, value: str, indent: int = 4, width: int = 26) -> None:
    pad = " " * (indent + width)
    lines = textwrap.wrap(value, WIDTH - width - indent + _ansi_len(value)) if len(value) > 60 else [value]
    print(" " * indent + dim(f"{key:<{width - 1}} ") + (lines[0] if lines else ""))
    for line in lines[1:]:
        print(pad + line)


def _ansi_len(text: str) -> int:
    """Number of invisible ANSI escape characters in ``text`` (so wrapping counts only visible ones)."""
    import re

    return sum(len(m) for m in re.findall(r"\033\[[0-9;]*m", text))


def scene(n: int, title: str) -> None:
    print()
    print(cyan("━" * WIDTH))
    print(cyan(bold(f" Scene {n} · {title}")))
    print(cyan("━" * WIDTH))


ACTION_STYLE = {"accept": green, "answer_from_evidence": green, "abstain": amber, "repair": amber,
                "regenerate": amber}


class Demo:
    def __init__(self) -> None:
        t0 = time.time()
        self.world = load_world()
        self.wmap = {w.id: w for w in self.world}
        self.nli = LexicalNLI()
        self.retriever = PassageRetriever(load_passages())
        self.corpus = GitaCorpus.load()
        self.threshold = tuned_support_threshold("lexical")
        self.verifier = VerseVerifier(self.corpus, support_threshold=self.threshold)
        self.policy = RepairPolicy(self.corpus)
        print(dim(f"  loaded {len(self.world)} QA-world questions, {len(self.corpus)} Gita verses, "
                  f"lexical NLI (support threshold {self.threshold:g}) in {time.time() - t0:.1f}s"))

    # ------------------------------------------------------------------ UG-RAV helpers
    def run_ugrav(self, question: str, aliases: list[str] | None = None) -> dict:
        # A fresh simulator per question keeps every scene reproducible on its own (--scene N).
        pipe = UGRAV(SimulatedQALLM(self.world, seed=0), self.nli, self.retriever, k=10)
        out = pipe.run(question, aliases=aliases)
        c, th, d = out["closed"], out["thresholds"], out["decision"]
        print()
        kv("question", bold(question))
        kv("closed-book answer", f"{c['greedy']!r}  " + self._mark(d.get("closed_correct")))
        kv("K=10 samples", textwrap.shorten(", ".join(s.rstrip(".") for s in c["samples"]), 140))
        kv("semantic entropy", f"{c['se']:.2f}  ({c['n_clusters']} meaning cluster(s); "
                               f"self-consistency {c['self_consistency']:.2f})")
        r0 = out["steps"][0]["risk_r0"]
        gate = "retrieve" if d["retrieved"] else "skip retrieval"
        kv("closed-book risk r0", f"{r0:.2f}  vs τ_low = {th['tau_low']:.2f}  →  {gate}")
        if out["rag"]:
            r = out["rag"]
            for p in r["passages"][:2]:
                kv("evidence", dim(p))
            kv("RAG answer", f"{r['greedy']!r}  " + self._mark(d.get("rag_correct")))
            kv("NLI support / ΔSE", f"{r['support_rag']:.2f} / {r['delta_se']:+.2f}  "
                                    f"(agreement with closed book: {'yes' if r['agreement'] else 'no'})")
            kv("fused risk r", f"{out['steps'][1]['risk_r_hat']:.2f}  vs τ_high = {th['tau_high']:.2f}")
        style = ACTION_STYLE.get(d["action"], bold)
        kv("decision", style(bold(d["action"].upper())) + f"  →  {d['answer']!r}  " + self._mark(
            d.get("final_correct")))
        kv("why", dim(d["rationale"]))
        return out

    @staticmethod
    def _mark(ok: bool | None) -> str:
        if ok is None:
            return ""
        return green("✓ correct") if ok else red("✗ wrong")

    # ------------------------------------------------------------------ VGHD helpers
    def run_verify(self, question: str, answer: str) -> None:
        rep = self.verifier.verify(question, answer)
        dec = self.policy.decide(rep)
        print()
        kv("question", bold(question))
        kv("answer", answer)
        verdict = red(bold("HALLUCINATED")) if rep.hallucinated else green(bold(rep.verdict.upper()))
        kv("verdict", f"{verdict}  (risk {rep.score:.2f})")
        for issue in rep.issues:
            kv(f"· {issue.type.value.replace('_', ' ')}", issue.message)
        style = ACTION_STYLE.get(dec.action, bold)
        kv("decision", style(bold(dec.action.upper())))
        if dec.action == "repair":
            after = self.verifier.verify(question, dec.answer)
            kv("repaired answer", dec.answer)
            kv("re-verified", green("faithful ✓") if not after.hallucinated else red("still hallucinated"))
        elif dec.action == "abstain":
            kv("response", dec.answer)

    # ------------------------------------------------------------------ scenes
    def scene1(self) -> None:
        scene(1, "The problem: fluent, confident and wrong")
        say("Ask a model “Who wrote Godan?” and it may answer “Rabindranath Tagore” — a perfect sentence with the "
            "wrong author (it is Munshi Premchand). Nothing in the text reveals the error, so we sample several "
            "answers and measure how much their *meanings* disagree.")
        answers = ["Premchand", "Munshi Premchand", "Rabindranath Tagore", "It was Premchand.", "Tagore",
                   "Premchand"]
        cl = cluster_answers(answers, self.nli, "Who wrote Godan?")
        print()
        for i, c in enumerate(cl.clusters, 1):
            kv(f"meaning cluster {i}", " | ".join(answers[j] for j in c))
        kv("lexical entropy", f"{lexical_entropy(answers):.2f}  (counts every spelling as different)")
        kv("semantic entropy", f"{semantic_entropy(cl):.2f}  (paraphrases merged by bidirectional entailment)")
        say(dim("High semantic entropy flags confabulation — but a model that is *consistently* wrong has low "
                "entropy. That blind spot is why UG-RAV adds retrieved, NLI-verified evidence."), indent=4)

    def scene2(self) -> None:
        scene(2, "UG-RAV: uncertainty opens the gate, evidence decides (Godan)")
        say("The closed-book samples disagree, so the gate retrieves passages; NLI checks that the evidence entails "
            "the evidence-grounded answer, and the fused risk is low enough to answer from evidence.")
        self.run_ugrav("Who wrote the Hindi novel Godan?", self.wmap["w001"].aliases)

    def scene3(self) -> None:
        scene(3, "A confident error that semantic entropy cannot see (crescograph)")
        say("All ten samples agree on the wrong inventor, so semantic entropy is 0 — an uncertainty-only detector "
            "would accept it. UG-RAV's fused closed-book risk still opens the gate, the evidence contradicts the "
            "closed-book answer, and the system answers from verified evidence.")
        item = self.wmap["w117"]
        self.run_ugrav(item.question, item.aliases)
        sim = _load("results/ugrav/benchmark_sim-7b.json")
        residue = (f"the {100 * sim['policies_test']['ug_rav']['hallucination_rate']:.1f} % hallucination rate "
                   "left on the simulation test split") if sim else "the residual hallucination rate"
        say(dim("Honest limitation: not every confident error is caught. “Who wrote the autobiography Wings of "
                "Fire?” is answered “C. V. Raman” consistently, its fused risk stays below τ_low and the wrong "
                f"answer is accepted. Errors of this kind are behind {residue}."), indent=4)

    def scene4(self) -> None:
        scene(4, "Accept when safe, abstain when nothing helps")
        say("Retrieval costs latency, so a well-known fact answered consistently is accepted without retrieval …")
        item = self.wmap["w003"]
        self.run_ugrav(item.question, item.aliases)
        say("… and when neither the model nor the evidence can answer, UG-RAV says “I don't know” instead of "
            "guessing.")
        self.run_ugrav("What is the capital of Mars?")

    def scene5(self) -> None:
        scene(5, "VGHD: verse-grounded verification of a Bhagavad Gita answer")
        say("The motivating example from the Part-I report: the claim is true, but the soul-and-weapons teaching "
            "is BG 2.23, not BG 2.20. Answer-level checks miss this; verse-level checks catch and repair it.")
        q = "Which verse of the Bhagavad Gita says that the soul cannot be cut by weapons?"
        self.run_verify(q, "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is "
                           "described in BG 2.20.")

    def scene6(self) -> None:
        scene(6, "VGHD catalogue: speaker, quotation, premise and scope")
        self.run_verify("What does Krishna say about action and its fruits?",
                        "Arjuna teaches that one's business is with the action only, never with its fruits "
                        "(BG 2.47).")
        self.run_verify("What does Krishna promise to those who take refuge in him?",
                        "Krishna says, \"Abandon all duties and come to me; I will make you rich and powerful\" "
                        "(BG 18.66).")
        self.run_verify("Why does Arjuna say \"Time am I, laying desolate the world\"?",
                        "Arjuna says these words because he is terrified by the universal form (BG 11.32).")
        self.run_verify("What does the Bhagavad Gita say about using smartphones?",
                        "The Gita advises that one should limit smartphone use to keep the mind steady (BG 6.26).")

    def scene7(self) -> None:
        scene(7, "Gita RAG assistant with an injected fabricated citation")
        say("Retrieve verses → generate a cited answer (simulator injects a fabricated verse number) → verify → "
            "repair → re-verify.")
        qa = GitaQA(self.corpus, verifier=self.verifier, simulator_mode="fabricated_citation")
        out = qa.ask("What does Krishna say about the right to action?")
        print()
        kv("retrieved", ", ".join(f"BG {h['ref']}" for h in out["retrieved"]))
        kv("generated", out["answer"])
        kv("detected", red(", ".join(out["report"]["types"]) or "nothing"))
        kv("decision", amber(bold(out["decision"]["action"].upper())))
        kv("final answer", out["decision"]["answer"])
        ok = not out["final_report"]["hallucinated"]
        kv("re-verified", green("faithful ✓") if ok else red("still hallucinated"))
        kv("latency", f"{out['seconds']:.2f}s on CPU")

    def scene8(self) -> None:
        scene(8, "Headline results (from results/*.json)")
        sim = _load("results/ugrav/benchmark_sim-7b.json")
        if sim:
            det, pol = sim["detection"], sim["policies_test"]
            say(bold("UG-RAV — simulation benchmark") + dim(" (phenomena encoded by construction; validates the "
                                                             "pipeline, not the hypothesis)"))
            kv("AUROC fused + evidence", f"{det['fused_with_evidence']['auroc']:.3f}   vs semantic entropy "
                                         f"{det['semantic_entropy']['auroc']:.3f}")
            for name in ("never_retrieve", "always_retrieve", "se_gated_no_verification", "ug_rav"):
                p = pol[name]
                kv(name, f"hallucination {p['hallucination_rate']:.3f} · coverage {p['coverage']:.3f} · "
                         f"retrieval {p['retrieval_rate']:.2f}")
        real = _load("results/real/benchmark_popqa_qwen2.5-1.5b-instruct.json")
        if real:
            det = real["detection"]
            say(bold("UG-RAV — real pilot") + dim(f" (Qwen2.5-1.5B-Instruct on CPU, PopQA, "
                                                  f"N = {real['n_questions']})"))
            kv("closed-book / RAG acc.", f"{real['closed_book_accuracy']:.3f} / {real['rag_accuracy']:.3f}")
            kv("AUROC SE / evidence", f"{det['semantic_entropy']['auroc']:.3f} / "
                                      f"{det['one_minus_evidence_support']['auroc']:.3f}")
        for tag, label in (("lexical", "lexical NLI"), ("microsoft_deberta_large_mnli", "DeBERTa-large-MNLI")):
            g = _load(f"results/gita/benchmark_{tag}.json")
            if g:
                v, a = g["detectors"]["vghd"], g["detectors"]["answer_level_nli"]
                say(bold(f"VGHD — {label}") + dim(" (synthetic hallucination test set, test split)"))
                kv("VGHD", f"AUROC {v['auroc']:.3f} · F1 {v['f1']:.3f} · FPR {v['false_positive_rate']:.3f}")
                kv("answer-level baseline", f"F1 {a['f1']:.3f} · FPR {a['false_positive_rate']:.3f}")
        say(dim("Full numbers, CIs, ablations and caveats: reports/03 and reports/04, or the Results pages of the "
                "Streamlit app (streamlit run app/Home.py)."), indent=4)


def _load(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    global COLOR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, choices=range(1, 9), help="run a single scene (1-8)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between scenes")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    args = ap.parse_args()
    COLOR = not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if COLOR and os.name == "nt":
        os.system("")  # enables ANSI escape sequences in the Windows console
    print(bold("\n  halludetect · Hallucination detection & mitigation demo (UG-RAV + VGHD)"))
    demo = Demo()
    scenes = [args.scene] if args.scene else list(range(1, 9))
    for i, n in enumerate(scenes):
        getattr(demo, f"scene{n}")()
        if args.pause and i < len(scenes) - 1:
            input(dim("\n  ⏎  press Enter for the next scene … "))
    print()


if __name__ == "__main__":
    main()
