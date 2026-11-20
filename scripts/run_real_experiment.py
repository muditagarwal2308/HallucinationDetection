"""Real-model UG-RAV experiment: PopQA + live Wikipedia retrieval + an open LLM + DeBERTa NLI.

This is a small-scale, CPU-friendly instance of the dissertation's experimental
pipeline. It is meant to validate the pipeline end to end on real model
behaviour, not to replace the planned 7–8B GPU runs (see the Kaggle notebook in
``notebooks/``).

Steps (every expensive step is cached, so the script can be resumed):

1. sample PopQA questions stratified by subject popularity (s_pop), skipping the
   open-ended ``occupation`` and ``genre`` relations whose gold alias lists are
   incomplete;
2. retrieve the top-3 Wikipedia articles for the question text with the
   Wikipedia search API and use the first ~80 words of each lead section as
   passages (a real, imperfect retriever);
3. generate 1 greedy + K sampled answers closed-book and with passages using a
   local Hugging Face model, recording token log-probabilities and P(True);
4. cluster answers and verify evidence with ``microsoft/deberta-large-mnli``;
5. evaluate signals, fusion and policies with ``halludetect.ugrav.benchmark.evaluate``.

Usage::

    python scripts/run_real_experiment.py --n 120 --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from halludetect.ugrav.benchmark import detection_markdown, evaluate, policy_markdown  # noqa: E402
from halludetect.ugrav.signals import QuestionRecord, build_record  # noqa: E402

UA = {"User-Agent": "HallucinationDetection-research/1.0 (https://github.com/muditagarwal2308/HallucinationDetection)"}
SKIP_RELATIONS = {"occupation", "genre"}


def get_json(url: str, params: dict, tries: int = 6) -> dict:
    """GET with exponential back-off (public APIs occasionally return 5xx / 429)."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            if r.status_code < 400:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {tries} attempts")


def fetch_popqa(cache: Path) -> list[dict]:
    """Download PopQA (Mallen et al., 2023; 14,267 questions) from the Hugging Face Hub and cache it."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("akariasai/PopQA", "test.tsv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["s_pop"] = int(r["s_pop"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rows = [r for r in rows if r["prop"] not in SKIP_RELATIONS]
    tiers = {"low": [], "mid": [], "high": []}
    for r in rows:
        p = r["s_pop"]
        tiers["low" if p < 300 else "mid" if p < 10000 else "high"].append(r)
    rng = random.Random(seed)
    picked = {}
    for name, items in tiers.items():
        rng.shuffle(items)
        picked[name] = [dict(r, pop_tier=name) for r in items[: n // 3]]
    # interleave the tiers so that a partial run is still balanced
    return [picked[t][i] for i in range(n // 3) for t in ("low", "mid", "high")]


class WikipediaRetriever:
    """Top-k Wikipedia articles for a query (search API) → first ~80 words of each lead section."""

    def __init__(self, cache_path: Path, words: int = 80):
        self.cache_path = cache_path
        self.words = words
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8")

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        key = f"{k}|{query}"
        if key not in self.cache:
            s = get_json("https://en.wikipedia.org/w/api.php", {
                "action": "query", "list": "search", "srsearch": query, "srlimit": k, "format": "json"})
            titles = [x["title"] for x in s.get("query", {}).get("search", [])]
            hits = []
            if titles:
                e = get_json("https://en.wikipedia.org/w/api.php", {
                    "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                    "titles": "|".join(titles), "format": "json"})
                pages = {p.get("title"): p.get("extract", "") for p in e.get("query", {}).get("pages", {}).values()}
                for i, t in enumerate(titles):
                    text = " ".join((pages.get(t) or "").split()[: self.words])
                    hits.append([t, f"{t}: {text}", float(k - i)])
            self.cache[key] = hits
            self._save()
            time.sleep(1.0)
        return [tuple(h) for h in self.cache[key]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=10, help="number of sampled answers")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--nli", default="microsoft/deberta-large-mnli")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "real")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes 4-bit quantisation (CUDA only)")
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    from halludetect.llm.hf_local import HFLocalLLM
    from halludetect.nli import get_nli

    tag = args.model.split("/")[-1].lower()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = stratified_sample(fetch_popqa(ROOT / "data" / "raw" / "popqa_test.json"), args.n, args.seed)
    retriever = WikipediaRetriever(ROOT / "data" / "real" / "wikipedia_passages.json")
    rec_path = out / f"records_popqa_{tag}.jsonl"
    done = {}
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["id"]] = d
    print(f"{len(rows)} questions; {len(done)} cached", flush=True)
    llm = HFLocalLLM(args.model, seed=args.seed, load_in_4bit=args.load_in_4bit)
    nli = get_nli(args.nli)
    t0 = time.time()
    with rec_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            qid = f"pq{row['id']}"
            if qid in done:
                continue
            aliases = json.loads(row["possible_answers"])
            meta = {"tier": row["pop_tier"], "popularity": row["s_pop"], "relation": row["prop"],
                    "subject": row["subj"], "model": args.model, "nli": args.nli}
            rec = build_record(qid, row["question"], aliases, llm, nli, retriever, k=args.k,
                               temperature=args.temperature, top_k=3, meta=meta)
            rec.meta["gold_retrieved"] = any(row["s_wiki_title"].lower() == pid.lower() for pid in rec.rag.passage_ids)
            rec.meta["answer_in_passages"] = any(a.lower() in " ".join(rec.rag.passages).lower() for a in aliases)
            fh.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            el = time.time() - t0
            ok_c = "✓" if rec.closed_correct else "✗"
            ok_r = "✓" if rec.rag_correct else "✗"
            print(f"[{i + 1}/{len(rows)}] {el / 60:5.1f} min  {row['question'][:60]!r:64} "
                  f"closed={rec.closed.greedy[:24]!r} ({ok_c}) rag={rec.rag.greedy[:24]!r} ({ok_r}) "
                  f"SE={rec.closed.se:.2f}", flush=True)
    records = [QuestionRecord.from_dict(json.loads(x)) for x in rec_path.read_text(encoding="utf-8").splitlines() if x]
    res = evaluate(records, label=f"real:popqa:{args.model}:{args.nli}:K={args.k}")
    res["answer_in_passages_rate"] = sum(r.meta.get("answer_in_passages", False) for r in records) / len(records)
    res["mean_seconds_per_question"] = sum(r.seconds for r in records) / len(records)
    (out / f"benchmark_popqa_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(detection_markdown(res))
    print(policy_markdown(res))


if __name__ == "__main__":
    main()
