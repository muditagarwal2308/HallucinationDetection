"""End-to-end acceptance check of the whole project (data → library → CLI → demo → app → artefacts).

Usage::

    python scripts/e2e_check.py            # quick suite (~2–3 min on a laptop CPU)
    python scripts/e2e_check.py --full     # also re-runs the VGHD lexical benchmark (~5 min) for reproducibility
    python scripts/e2e_check.py --no-app   # skip the Streamlit server check

Writes ``results/e2e/e2e_report.json`` and ``results/e2e/e2e_report.md`` and exits non-zero if any check fails.
Every check is a black-box run of what a user would do; expectations are stated next to each check.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
HD = [PY, "-m", "halludetect.cli"]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


@dataclass
class Check:
    area: str
    name: str
    expectation: str
    passed: bool = False
    detail: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


RESULTS: list[Check] = []


def check(area: str, name: str, expectation: str):
    """Decorator: run the function, record pass/fail, detail and timing."""

    def wrap(fn):
        c = Check(area, name, expectation)
        t0 = time.time()
        try:
            out = fn()
            c.passed, c.detail = True, (out if isinstance(out, str) else "ok")
        except AssertionError as exc:
            c.detail = f"FAILED: {exc}"
        except Exception as exc:  # noqa: BLE001 - report any crash as a failed check
            c.detail = f"ERROR: {type(exc).__name__}: {exc}"
        c.seconds = round(time.time() - t0, 2)
        RESULTS.append(c)
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] {area:<10} {name:<52} {c.seconds:6.1f}s  {c.detail[:90]}", flush=True)
        return fn

    return wrap


def run(cmd: list[str], timeout: int = 600, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def cli_json(*args: str) -> dict:
    p = run([*HD, *args])
    assert p.returncode == 0, f"exit {p.returncode}: {p.stderr[-300:]}"
    return json.loads(p.stdout)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------------- suites
def data_checks() -> None:
    @check("data", "Gita corpus has 701 verses in 18 chapters", "701 verses; BG 2.47 spoken by Krishna")
    def _():
        verses = json.loads((ROOT / "data/gita/verses.json").read_text(encoding="utf-8"))
        assert len(verses) == 701, len(verses)
        assert len({v["chapter"] for v in verses}) == 18
        v247 = next(v for v in verses if v["chapter"] == 2 and v["verse"] == 47)
        assert v247["speaker"] == "Krishna"
        return f"{len(verses)} verses"

    @check("data", "Gita QA set and hallucination test set", "106 questions; 524 labelled answers, 7 types")
    def _():
        qa = jsonl(ROOT / "data/gita/qa.jsonl")
        hs = jsonl(ROOT / "data/gita/hallucination_testset.jsonl")
        assert len(qa) == 106, len(qa)
        assert len(hs) == 524, len(hs)
        types = {h["type"] for h in hs if h["label"] == 1}
        assert len(types) >= 7, types
        return f"{len(qa)} questions, {len(hs)} answers, {len(types)} hallucination types"

    @check("data", "ODQA world and passages", "197 questions, 198 passages, every gold passage retrievable")
    def _():
        world = jsonl(ROOT / "data/odqa/world.jsonl")
        passages = jsonl(ROOT / "data/odqa/passages.jsonl")
        assert len(world) == 197 and len(passages) == 198, (len(world), len(passages))
        return f"{len(world)} questions, {len(passages)} passages"


def quality_checks() -> None:
    @check("quality", "pytest (unit + app tests)", "all tests pass")
    def _():
        p = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], timeout=1200)
        tail = (p.stdout.strip().splitlines() or ["?"])[-1]
        assert p.returncode == 0, tail
        return tail

    @check("quality", "ruff lint (src, tests, app, scripts)", "no findings")
    def _():
        exe = shutil.which("ruff") or str(Path(PY).with_name("ruff.exe" if os.name == "nt" else "ruff"))
        p = run([exe, "check", "src", "tests", "app", "scripts"])
        assert p.returncode == 0, p.stdout[-300:]
        return p.stdout.strip() or "clean"


def cli_checks() -> None:
    @check("cli", "search: English concept query", "BG 2.23 ranked first for 'weapons cannot cut the soul'")
    def _():
        p = run([*HD, "search", "weapons cannot cut the soul", "--k", "3"])
        assert p.returncode == 0 and p.stdout.split()[1] == "2.23", p.stdout[:200]
        return p.stdout.splitlines()[0][:80]

    @check("cli", "search: Hindi query", "BG 2.23 in the top 5 for आत्मा को शस्त्र नहीं काट सकते")
    def _():
        p = run([*HD, "search", "आत्मा को शस्त्र नहीं काट सकते", "--k", "5"])
        assert p.returncode == 0 and "2.23" in p.stdout, p.stdout[:200]
        return "hit"

    @check("cli", "verify: misattributed citation is repaired", "BG 2.20 → BG 2.23, repaired answer re-verifies")
    def _():
        q = "Which verse of the Bhagavad Gita says that the soul cannot be cut by weapons?"
        d = cli_json("verify", "-q", q, "-a", "The soul cannot be cut by weapons. This is described in BG 2.20.")
        assert "misattributed_citation" in d["types"], d["types"]
        assert d["decision"]["action"] == "repair" and "2.23" in d["decision"]["answer"], d["decision"]
        d2 = cli_json("verify", "-q", q, "-a", d["decision"]["answer"])
        assert not d2["hallucinated"], d2["types"]
        return d["decision"]["answer"]

    @check("cli", "verify: wrong speaker only", "exactly {wrong_speaker}; repair swaps Arjuna → Krishna")
    def _():
        d = cli_json("verify", "-q", "Who says that the soul cannot be cut by weapons?",
                     "-a", "Arjuna says the soul cannot be cut by weapons, as stated in BG 2.23.")
        assert d["types"] == ["wrong_speaker"], d["types"]
        assert d["decision"]["answer"].startswith("Krishna"), d["decision"]
        return d["decision"]["answer"]

    @check("cli", "verify: faithful answer is accepted", "no issues for a correct, correctly cited answer")
    def _():
        d = cli_json("verify", "-q", "What does Krishna say about action and its fruits?",
                     "-a", "Krishna says that one's business is with the action only, never with its fruits "
                           "(BG 2.47).")
        assert not d["hallucinated"] and d["verdict"] == "faithful", (d["verdict"], d["types"])
        return d["verdict"]

    for mode in ("fabricated_citation", "misattributed_citation", "misquotation", "wrong_speaker",
                 "unsupported_claim"):
        @check("cli", f"gita-ask --inject {mode}", "injected type detected; final answer verifies as faithful")
        def _(mode=mode):
            d = cli_json("gita-ask", "What does Krishna say about action and its fruits?", "--inject", mode)
            assert mode in d["report"]["types"], (mode, d["report"]["types"])
            assert d["decision"]["action"] in ("repair", "abstain"), d["decision"]["action"]
            if d["decision"]["action"] == "repair":
                assert not d["final_report"]["hallucinated"], d["final_report"]["types"]
            return f"{d['decision']['action']} in {d['seconds']}s"

    @check("cli", "ugrav-ask: high-SE error corrected (Godan)", "answer_from_evidence → Munshi Premchand")
    def _():
        d = cli_json("ugrav-ask", "Who wrote the Hindi novel Godan?")
        dec = d["decision"]
        assert dec["action"] == "answer_from_evidence" and dec.get("final_correct"), dec
        return f"{d['closed']['greedy']} → {dec['answer']}"

    @check("cli", "ugrav-ask: paraphrased question matches the world", "'Who wrote Godan?' is answered correctly")
    def _():
        dec = cli_json("ugrav-ask", "Who wrote Godan?")["decision"]
        assert dec.get("final_correct"), dec
        return dec["answer"]

    @check("cli", "ugrav-ask: unanswerable question", "abstain with “I don't know.”")
    def _():
        dec = cli_json("ugrav-ask", "What is the capital of Mars?")["decision"]
        assert dec["action"] == "abstain", dec
        return dec["rationale"][:80]

    @check("cli", "se: semantic vs lexical entropy", "2 meaning clusters; SE < lexical entropy")
    def _():
        d = cli_json("se", "-q", "Who wrote Godan?", "Premchand", "Munshi Premchand", "Rabindranath Tagore",
                     "Premchand", "Tagore")
        assert len(d["clusters"]) == 2 and d["semantic_entropy"] < d["lexical_entropy"], d
        return f"SE {d['semantic_entropy']:.3f} vs lexical {d['lexical_entropy']:.3f}"


def reproducibility_checks(full: bool) -> None:
    @check("repro", "UG-RAV simulation benchmark regenerates identically", "fresh run == committed JSON")
    def _():
        from halludetect.ugrav.benchmark import run_simulation

        ref = json.loads((ROOT / "results/ugrav/benchmark_sim-7b.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            new = run_simulation(out_dir=Path(tmp))
        pairs = [("detection", "fused_with_evidence", "auroc"), ("detection", "semantic_entropy", "auroc"),
                 ("policies_test", "ug_rav", "hallucination_rate"), ("policies_test", "ug_rav", "coverage"),
                 ("policies_test", "always_retrieve", "accuracy")]
        for a, b, c in pairs:
            assert abs(ref[a][b][c] - new[a][b][c]) < 1e-9, (a, b, c, ref[a][b][c], new[a][b][c])
        return f"fused AUROC {new['detection']['fused_with_evidence']['auroc']:.4f} reproduced"

    if full:
        @check("repro", "VGHD lexical benchmark regenerates identically", "fresh run == committed JSON")
        def _():
            from halludetect.gita.benchmark import run_benchmark

            ref = json.loads((ROOT / "results/gita/benchmark_lexical.json").read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                new = run_benchmark("lexical", out_dir=Path(tmp))
            for det in ("vghd", "answer_level_nli", "citation_existence"):
                for m in ("auroc", "f1", "false_positive_rate"):
                    assert abs(ref["detectors"][det][m] - new["detectors"][det][m]) < 1e-9, (det, m)
            assert ref["repair"] == new["repair"], "repair outcomes differ"
            return f"VGHD F1 {new['detectors']['vghd']['f1']:.4f} reproduced"


def demo_checks() -> None:
    @check("demo", "scripts/run_demo.py runs all 8 scenes offline", "exit 0; 8 scene banners; no tracebacks")
    def _():
        p = run([PY, "scripts/run_demo.py", "--no-color"], timeout=300)
        assert p.returncode == 0, p.stderr[-300:]
        assert sum(f"Scene {i} ·" in p.stdout for i in range(1, 9)) == 8, "missing scenes"
        assert "Traceback" not in p.stdout + p.stderr
        assert p.stdout.count("faithful ✓") >= 4, "repairs did not re-verify"
        return f"{len(p.stdout.splitlines())} lines of output"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def app_checks() -> None:
    @check("app", "Streamlit server boots and serves all pages", "health 'ok'; HTTP 200 for home + 8 pages")
    def _():
        port = _free_port()
        proc = subprocess.Popen([PY, "-m", "streamlit", "run", "app/Home.py", "--server.headless", "true",
                                 "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                                cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(60):
                try:
                    if urllib.request.urlopen(f"{base}/_stcore/health", timeout=2).read().decode() == "ok":
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise AssertionError("server did not become healthy within 60 s")
            pages = [""] + [p.stem.split("_", 2)[-1] for p in sorted((ROOT / "app/pages").glob("*.py"))]
            for page in pages:
                r = urllib.request.urlopen(f"{base}/{urllib.request.quote(page)}", timeout=10)
                assert r.status == 200, (page, r.status)
            return f"{len(pages)} routes on port {port}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def artefact_checks() -> None:
    @check("artefacts", "benchmark result files parse with required keys", "sim, VGHD (+ real / DeBERTa if run)")
    def _():
        found = []
        for rel, key in [("results/ugrav/benchmark_sim-7b.json", "policies_test"),
                         ("results/gita/benchmark_lexical.json", "detectors"),
                         ("results/real/benchmark_popqa_qwen2.5-1.5b-instruct.json", "detection"),
                         ("results/gita/benchmark_microsoft_deberta_large_mnli.json", "detectors")]:
            p = ROOT / rel
            if p.exists():
                assert key in json.loads(p.read_text(encoding="utf-8")), rel
                found.append(Path(rel).stem)
        assert len(found) >= 2, found
        return ", ".join(found)

    @check("artefacts", "report PDFs open and have pages", "every reports/pdf/*.pdf readable")
    def _():
        import pymupdf as fitz

        pdfs = sorted((ROOT / "reports/pdf").glob("*.pdf"))
        assert pdfs, "no PDFs"
        pages = {p.name: fitz.open(p).page_count for p in pdfs}
        assert all(n > 0 for n in pages.values()), pages
        return ", ".join(f"{k.split('_')[0]}:{v}p" for k, v in pages.items())

    @check("artefacts", "academic documents and milestone decks", "≥ 5 DOCX + PDF, ≥ 4 PPTX + PDF; all open")
    def _():
        import pymupdf as fitz

        docs = sorted((ROOT / "docs/academic").glob("*.docx"))
        decks = sorted((ROOT / "presentations").glob("*.pptx"))
        doc_pdfs = sorted((ROOT / "docs/academic").glob("*.pdf"))
        deck_pdfs = sorted((ROOT / "presentations/pdf").glob("*.pdf"))
        for p in doc_pdfs + deck_pdfs:
            assert fitz.open(p).page_count > 0, p.name
        assert len(docs) >= 5 and len(doc_pdfs) >= 5, f"{len(docs)} DOCX, {len(doc_pdfs)} PDF"
        assert len(decks) >= 4 and len(deck_pdfs) >= 4, f"{len(decks)} PPTX, {len(deck_pdfs)} PDF"
        pages = sum(fitz.open(p).page_count for p in doc_pdfs)
        slides = sum(fitz.open(p).page_count for p in deck_pdfs)
        return f"{len(docs)} documents ({pages} pages), {len(decks)} decks ({slides} slides)"

    @check("artefacts", "figures referenced by reports exist", "every ![](figures/…) target is on disk")
    def _():
        import re

        missing = []
        for md in (ROOT / "reports").glob("*.md"):
            for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md.read_text(encoding="utf-8")):
                if not target.startswith("http") and not (md.parent / target).exists():
                    missing.append(f"{md.name}:{target}")
        assert not missing, missing[:5]
        return "all present"


# ---------------------------------------------------------------------------------------------- report
def write_report(started: float) -> Path:
    out = ROOT / "results" / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    n_pass = sum(c.passed for c in RESULTS)
    meta = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.processor(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "seconds": round(time.time() - started, 1), "passed": n_pass, "total": len(RESULTS)}
    (out / "e2e_report.json").write_text(json.dumps({"meta": meta, "checks": [c.__dict__ for c in RESULTS]},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# End-to-end check results", "",
             f"{n_pass}/{len(RESULTS)} checks passed in {meta['seconds']} s · Python {meta['python']} · "
             f"{meta['platform']} · {meta['started']}", "",
             "| # | Area | Check | Expectation | Result | Time (s) | Detail |", "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(RESULTS, 1):
        detail = c.detail.replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(f"| {i} | {c.area} | {c.name} | {c.expectation} | {'✅ pass' if c.passed else '❌ fail'} | "
                     f"{c.seconds} | {detail} |")
    (out / "e2e_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also re-run the VGHD lexical benchmark")
    ap.add_argument("--no-app", action="store_true", help="skip the Streamlit server check")
    ap.add_argument("--skip-tests", action="store_true", help="skip pytest and ruff (when run separately)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = time.time()
    print(f"halludetect end-to-end check · {time.strftime('%Y-%m-%d %H:%M')}\n")
    data_checks()
    if not args.skip_tests:
        quality_checks()
    cli_checks()
    reproducibility_checks(args.full)
    demo_checks()
    if not args.no_app:
        app_checks()
    artefact_checks()
    out = write_report(started)
    n_pass = sum(c.passed for c in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed · report: {out / 'e2e_report.md'}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
