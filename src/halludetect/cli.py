"""Command-line interface: ``halludetect <command>``.

Commands
--------
verify        verify a Gita answer verse by verse (and repair or abstain)
gita-ask      retrieval-augmented Gita QA with verification (simulator or any OpenAI-compatible LLM)
ugrav-ask     run UG-RAV on one question (simulator world or a real LLM)
se            semantic entropy of a set of answers
bench-gita    hallucination-detection benchmark on the Gita test set
bench-ugrav   simulated UG-RAV benchmark (signals, fusion, policies, popularity)
search        search the Gita corpus (English, Hindi or romanised Sanskrit)
"""

from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _llm(args):
    if getattr(args, "llm", None):
        from .llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(model=args.llm, base_url=args.base_url)
    return None


def cmd_verify(args) -> None:
    from .gita import GitaCorpus, VerseVerifier
    from .gita.policy import RepairPolicy
    from .nli import get_nli

    corpus = GitaCorpus.load()
    nli = None if args.nli == "lexical" else get_nli(args.nli)
    v = VerseVerifier(corpus, nli)
    rep = v.verify(args.question, args.answer)
    out = rep.as_dict()
    out["decision"] = RepairPolicy(corpus).decide(rep).as_dict()
    _print(out)


def cmd_gita_ask(args) -> None:
    from .gita.pipeline import GitaQA

    qa = GitaQA(llm=_llm(args), simulator_mode=args.inject)
    _print(qa.ask(args.question))


def cmd_ugrav_ask(args) -> None:
    from .nli import get_nli
    from .ugrav import UGRAV, load_passages, load_world
    from .ugrav.data import PassageRetriever
    from .ugrav.simulator import SimulatedQALLM

    world = load_world()
    llm = _llm(args) or SimulatedQALLM(world, seed=args.seed)
    nli = get_nli(args.nli)
    item = next((w for w in world if w.question.lower() == args.question.lower()), None)
    pipe = UGRAV(llm, nli, PassageRetriever(load_passages()), k=args.k)
    _print(pipe.run(args.question, aliases=item.aliases if item else None))


def cmd_se(args) -> None:
    from .nli import get_nli
    from .uncertainty import cluster_answers, lexical_entropy, max_entropy, semantic_entropy

    nli = get_nli(args.nli)
    cl = cluster_answers(args.answers, nli, args.question)
    _print({"clusters": [[args.answers[i] for i in c] for c in cl.clusters], "semantic_entropy": semantic_entropy(cl),
            "lexical_entropy": lexical_entropy(args.answers), "max_entropy": max_entropy(len(args.answers))})


def cmd_bench_gita(args) -> None:
    from .gita.benchmark import markdown_table, run_benchmark

    res = run_benchmark(args.nli, search_k=args.search_k)
    print(markdown_table(res))


def cmd_bench_ugrav(args) -> None:
    from .ugrav.benchmark import detection_markdown, run_simulation

    seeds = tuple(range(args.seeds))
    res = run_simulation(seeds=seeds, profile=args.profile, verbose=False)
    print(detection_markdown(res))
    from .ugrav.benchmark import policy_markdown

    print()
    print(policy_markdown(res))


def cmd_search(args) -> None:
    from .gita import GitaCorpus

    corpus = GitaCorpus.load()
    for v, s in corpus.search(args.query, k=args.k):
        print(f"BG {v.ref:6s} {s:6.3f}  [{v.speaker}] {v.english}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="halludetect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("verify", help="verify a Gita answer verse by verse")
    s.add_argument("--question", "-q", required=True)
    s.add_argument("--answer", "-a", required=True)
    s.add_argument("--nli", default="lexical", help="'lexical' or a Hugging Face NLI model id")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("gita-ask", help="retrieval-augmented Gita QA with verification")
    s.add_argument("question")
    s.add_argument("--inject", default="none", help="simulator only: none, fabricated_citation, "
                   "misattributed_citation, misquotation, wrong_speaker, unsupported_claim, random")
    s.add_argument("--llm", help="model name on an OpenAI-compatible server, e.g. qwen2.5:7b (Ollama)")
    s.add_argument("--base-url", default="http://localhost:11434/v1")
    s.set_defaults(func=cmd_gita_ask)

    s = sub.add_parser("ugrav-ask", help="run UG-RAV on one question")
    s.add_argument("question")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--nli", default="lexical")
    s.add_argument("--llm")
    s.add_argument("--base-url", default="http://localhost:11434/v1")
    s.set_defaults(func=cmd_ugrav_ask)

    s = sub.add_parser("se", help="semantic entropy of answers")
    s.add_argument("--question", "-q", default="")
    s.add_argument("--nli", default="lexical")
    s.add_argument("answers", nargs="+")
    s.set_defaults(func=cmd_se)

    s = sub.add_parser("bench-gita", help="VGHD benchmark")
    s.add_argument("--nli", default="lexical")
    s.add_argument("--search-k", type=int, default=12)
    s.set_defaults(func=cmd_bench_gita)

    s = sub.add_parser("bench-ugrav", help="simulated UG-RAV benchmark")
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--profile", default="sim-7b")
    s.set_defaults(func=cmd_bench_ugrav)

    s = sub.add_parser("search", help="search the Gita corpus")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.set_defaults(func=cmd_search)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
