"""Benchmark for verse-level hallucination detection (dissertation Objective O2 of the Gita study).

Detectors compared on ``data/gita/hallucination_testset.jsonl``:

* ``citation_existence`` – flags only references to verses that do not exist;
* ``answer_level_nli``   – RAGAS/ALCE-style faithfulness of the answer to the
  verses retrieved for the question, ignoring which verse each claim cites
  (what a general-domain RAG faithfulness checker sees);
* ``vghd``               – the proposed verse-grounded detector, plus ablations
  without the premise, quotation or speaker checks.

The support threshold of VGHD and the decision threshold of the answer-level
baseline are tuned on the *dev* split (odd question numbers) by Youden's J
(TPR − FPR), and all headline numbers are reported on the held-out *test* split
(even question numbers).
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import metrics
from ..nli import get_nli
from ..nli.base import NLIModel
from ..paths import GITA_DATA, RESULTS
from .citations import extract_citations, strip_citations
from .claims import segment_claims
from .corpus import GitaCorpus
from .policy import RepairPolicy
from .verifier import GITA_LIGHT_TERMS, VerseVerifier

TYPES = ["fabricated_citation", "misattributed_citation", "misquotation", "wrong_speaker", "unsupported_claim",
         "false_premise_accepted", "out_of_scope_answered"]


def load_testset(path: Path | None = None) -> list[dict]:
    p = path or GITA_DATA / "hallucination_testset.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_of(qid: str) -> str:
    return "dev" if int(re.sub(r"\D", "", qid)) % 2 == 1 else "test"


@dataclass
class Prediction:
    score: float
    flagged: bool
    types: set[str]


class CitationExistenceBaseline:
    name = "citation_existence"

    def __init__(self, corpus: GitaCorpus):
        self.corpus = corpus

    def __call__(self, question: str, answer: str, threshold: float = 0.5) -> Prediction:
        bad = any(not self.corpus.exists(ch, v) for c in extract_citations(answer) for ch, v in c.refs)
        return Prediction(1.0 if bad else 0.0, bad, {"fabricated_citation"} if bad else set())


class AnswerLevelBaseline:
    """Faithfulness of each claim to the top-k verses retrieved for the *question* (citations ignored)."""

    name = "answer_level_nli"

    def __init__(self, corpus: GitaCorpus, nli: NLIModel, k: int = 5):
        self.corpus, self.nli, self.k = corpus, nli, k

    def score(self, question: str, answer: str) -> float:
        verses = [v for v, _ in self.corpus.search(question, k=self.k)]
        claims = [strip_citations(c.text) for c in segment_claims(answer)] or [answer]
        if not verses:
            return 1.0
        res = self.nli.predict_batch([(v.premise(), c) for c in claims for v in verses])
        step = len(verses)
        per_claim = [max(r.entailment for r in res[i * step:(i + 1) * step]) for i in range(len(claims))]
        return 1.0 - min(per_claim)

    def __call__(self, question: str, answer: str, threshold: float = 0.5) -> Prediction:
        s = self.score(question, answer)
        return Prediction(s, s >= threshold, {"unsupported_claim"} if s >= threshold else set())


def evaluate_retrieval(corpus: GitaCorpus, k: int = 5) -> dict:
    """Hit@1, recall@k and MRR of hybrid verse retrieval on the answerable QA questions, by language."""
    qa = [json.loads(x) for x in (GITA_DATA / "qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out: dict = {}
    for lang in ("all", "en", "hi"):
        rows = [q for q in qa if q["answerable"] and (lang == "all" or q["lang"] == lang)]
        hit1 = rec = rr = 0.0
        for q in rows:
            refs = [v.ref for v, _ in corpus.search(q["question"], k=k)]
            ranks = [i for i, r in enumerate(refs) if r in q["gold_verses"]]
            hit1 += bool(ranks) and ranks[0] == 0
            rec += bool(ranks)
            rr += 1 / (ranks[0] + 1) if ranks else 0.0
        n = max(len(rows), 1)
        out[lang] = {"n": len(rows), "hit_at_1": hit1 / n, f"recall_at_{k}": rec / n, "mrr": rr / n}
    return out


def _evaluate(rows: list[dict], preds: list[Prediction]) -> dict:
    y = np.array([r["label"] for r in rows])
    s = np.array([p.score for p in preds])
    f = np.array([p.flagged for p in preds])
    prec, rec, f1 = metrics.precision_recall_f1(f, y)
    neg = y == 0
    per_type = {}
    for t in TYPES:
        m = np.array([r["type"] == t for r in rows])
        if m.any():
            hit = [t in p.types for p, r in zip(preds, rows) if r["type"] == t]
            per_type[t] = {"n": int(m.sum()), "recall": float(f[m].mean()), "type_accuracy": float(np.mean(hit))}
    return {
        "n": int(len(rows)), "positives": int(y.sum()), "auroc": metrics.auroc(s, y),
        "auprc": metrics.auprc(s, y), "precision": prec, "recall": rec, "f1": f1,
        "false_positive_rate": float(f[neg].mean()) if neg.any() else float("nan"),
        "accuracy": float((f == y).mean()), "per_type": per_type,
    }


def _run(detector, rows: list[dict]) -> list[Prediction]:
    return [detector(r["question"], r["answer"]) for r in rows]


def _vghd_runner(verifier: VerseVerifier):
    def run(question: str, answer: str) -> Prediction:
        rep = verifier.verify(question, answer)
        return Prediction(rep.score, rep.hallucinated, rep.types)
    return run


def run_benchmark(nli_name: str = "lexical", out_dir: Path | None = None, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                  verbose: bool = True, search_k: int = 12) -> dict:
    t0 = time.time()
    corpus = GitaCorpus.load()
    if nli_name == "lexical":
        nli = get_nli("lexical", idf=corpus.idf, light_terms=GITA_LIGHT_TERMS)
    else:
        tag = re.sub(r"[^A-Za-z0-9]+", "_", nli_name).strip("_")
        nli = get_nli(nli_name, cache_path=(out_dir or RESULTS / "gita") / f"nli_cache_{tag}.json")
    rows = load_testset()
    dev = [r for r in rows if split_of(r["qid"]) == "dev"]
    test = [r for r in rows if split_of(r["qid"]) == "test"]

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    # ---- tune VGHD support threshold on dev
    tuning = {}
    for th in thresholds:
        v = VerseVerifier(corpus, nli, support_threshold=th, search_k=search_k)
        tuning[th] = _evaluate(dev, _run(_vghd_runner(v), dev))
        log(f"dev support_threshold={th}: F1={tuning[th]['f1']:.3f} TPR={tuning[th]['recall']:.3f} "
            f"FPR={tuning[th]['false_positive_rate']:.3f}")
    # Youden's J (TPR - FPR) is insensitive to the 4:1 class imbalance of the test set.
    best_th = max(thresholds, key=lambda t: (round(tuning[t]["recall"] - tuning[t]["false_positive_rate"], 4),
                                             tuning[t]["f1"]))

    # ---- tune answer-level baseline decision threshold on dev
    alb = AnswerLevelBaseline(corpus, nli)
    dev_scores = [alb.score(r["question"], r["answer"]) for r in dev]
    grid = np.linspace(0.05, 0.95, 19)
    y_dev = np.array([r["label"] for r in dev])
    best_alb, best_j = 0.5, -1.0
    for th in grid:
        flag = np.array(dev_scores) >= th
        j = flag[y_dev == 1].mean() - flag[y_dev == 0].mean()
        if j > best_j:
            best_alb, best_j = float(th), j
    log(f"tuned: VGHD support_threshold={best_th}, answer-level threshold={best_alb:.2f}")

    detectors = {
        "citation_existence": CitationExistenceBaseline(corpus),
        "answer_level_nli": lambda q, a: alb(q, a, threshold=best_alb),
        "vghd": _vghd_runner(VerseVerifier(corpus, nli, support_threshold=best_th, search_k=search_k)),
        "vghd_no_premise": _vghd_runner(VerseVerifier(corpus, nli, support_threshold=best_th, search_k=search_k,
                                                      check_premise=False)),
        "vghd_no_quotes": _vghd_runner(VerseVerifier(corpus, nli, support_threshold=best_th, search_k=search_k,
                                                     check_quotes=False)),
        "vghd_no_speaker": _vghd_runner(VerseVerifier(corpus, nli, support_threshold=best_th, search_k=search_k,
                                                      check_speaker=False)),
    }
    results: dict = {"nli": nli_name, "search_k": search_k, "support_threshold": best_th,
                     "answer_level_threshold": best_alb,
                     "tuning": {str(k): v for k, v in tuning.items()}, "detectors": {}}
    items = []
    for name, det in detectors.items():
        preds = _run(det, test)
        results["detectors"][name] = _evaluate(test, preds)
        log(f"test {name}: F1={results['detectors'][name]['f1']:.3f} AUROC={results['detectors'][name]['auroc']:.3f}")
        if name == "vghd":
            for r, p in zip(test, preds):
                items.append({"id": r["id"], "type": r["type"], "label": r["label"], "score": round(p.score, 4),
                              "flagged": p.flagged, "predicted_types": sorted(p.types)})
    # full-set numbers (dev + test) for reference
    full = VerseVerifier(corpus, nli, support_threshold=best_th, search_k=search_k)
    results["vghd_all_items"] = _evaluate(rows, _run(_vghd_runner(full), rows))

    # ---- repair / abstain effectiveness on the test split
    policy = RepairPolicy(corpus)
    outcome = defaultdict(lambda: defaultdict(int))
    examples = []
    for r in test:
        if r["label"] != 1:
            continue
        rep = full.verify(r["question"], r["answer"])
        if not rep.hallucinated:
            outcome[r["type"]]["missed"] += 1
            continue
        d = policy.decide(rep)
        if d.action == "abstain":
            outcome[r["type"]]["abstained"] += 1
            ok = r["type"] in ("false_premise_accepted", "out_of_scope_answered")
            outcome[r["type"]]["resolved"] += int(ok)
        else:
            rep2 = full.verify(r["question"], d.answer)
            fixed = not rep2.hallucinated
            outcome[r["type"]]["repaired_and_verified" if fixed else "repair_failed"] += 1
            outcome[r["type"]]["resolved"] += int(fixed)
            if len(examples) < 12 and fixed and d.edits:
                examples.append({"type": r["type"], "before": r["answer"], "after": d.answer, "edits": d.edits})
    results["repair"] = {t: dict(v) for t, v in outcome.items()}
    results["repair_examples"] = examples
    results["retrieval"] = evaluate_retrieval(corpus)
    getattr(nli, "save_cache", lambda: None)()
    results["runtime_seconds"] = round(time.time() - t0, 1)

    out = out_dir or RESULTS / "gita"
    out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", nli_name).strip("_")
    (out / f"benchmark_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / f"predictions_{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    log(f"wrote {out / f'benchmark_{tag}.json'}")
    return results


def markdown_table(results: dict) -> str:
    rows = ["| Detector | AUROC | Precision | Recall | F1 | FPR (correct answers) |", "|---|---|---|---|---|---|"]
    for name, r in results["detectors"].items():
        rows.append(f"| {name} | {r['auroc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                    f"{r['false_positive_rate']:.3f} |")
    return "\n".join(rows)
