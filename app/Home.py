"""Hallucination Detection Studio — entry point.

Run from the repository root::

    streamlit run app/Home.py

This file is the Home page; the other pages live in `app/pages/` (numbered so that they appear in order).
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import common as C  # noqa: E402
import streamlit as st  # noqa: E402

PAGES = {  # key: (path relative to this file, icon)
    "explorer": ("pages/1_🧩_Hallucination_Explorer.py", "🧩"),
    "se_lab": ("pages/2_🎲_Semantic_Entropy_Lab.py", "🎲"),
    "ugrav_live": ("pages/3_🧭_UG-RAV_Live.py", "🧭"),
    "ugrav_results": ("pages/4_📈_UG-RAV_Results.py", "📈"),
    "verifier": ("pages/5_📜_Gita_Verse_Verifier.py", "📜"),
    "rag": ("pages/6_🪔_Gita_RAG_Assistant.py", "🪔"),
    "vghd_results": ("pages/7_🏅_VGHD_Results.py", "🏅"),
    "corpus": ("pages/8_📖_Corpus_Browser.py", "📖"),
}

UGRAV_DOT = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.45; ranksep=0.28; pad=0.15;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#C7D2FE",
        fillcolor="#EEF2FF", fontcolor="#1E1B4B", penwidth=1.2, margin="0.16,0.07"];
  edge [color="#94A3B8", fontname="Helvetica", fontsize=11, fontcolor="#475569", arrowsize=0.7];
  q   [label="Question", shape=oval, fillcolor="#FFF7E6", color="#F59E0B"];
  cb  [label="Closed-book sampling: greedy a₀ + K samples"];
  sig [label="Semantic entropy · self-consistency · P(True)\\n→ fused closed-book risk r̂₀"];
  g1  [label="r̂₀ < τ_low ?", shape=diamond, fillcolor="#FFFBEB", color="#F59E0B"];
  acc [label="Accept a₀\\n(no retrieval)", fillcolor="#DCFCE7", color="#86EFAC"];
  ret [label="Retrieve top-3 passages · re-sample with evidence"];
  ev  [label="NLI support · contradiction · ΔSE · agreement\\n→ fused risk r̂"];
  g2  [label="r̂ < τ_high and\\nsupported ?", shape=diamond, fillcolor="#FFFBEB", color="#F59E0B"];
  ans [label="Answer from evidence", fillcolor="#E0E7FF", color="#A5B4FC"];
  abs [label="Abstain", fillcolor="#F1F5F9", color="#CBD5E1"];
  {rank=same; g1; acc;}
  q -> cb -> sig -> g1;
  g1 -> acc [label=" yes"];
  g1 -> ret [label=" no"];
  ret -> ev -> g2;
  g2 -> ans [label=" yes"];
  g2 -> abs [label=" no"];
}
"""

VGHD_DOT = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.28; pad=0.15;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#FCD34D",
        fillcolor="#FFFBEB", fontcolor="#1E1B4B", penwidth=1.2, margin="0.16,0.07"];
  edge [color="#94A3B8", fontname="Helvetica", fontsize=11, fontcolor="#475569", arrowsize=0.7];
  q    [label="Question (English · Hindi · romanised Sanskrit)", shape=oval, fillcolor="#EEF2FF",
        color="#A5B4FC"];
  ret  [label="Hybrid verse retrieval over 701 verses"];
  gen  [label="LLM answer with BG chapter.verse citations"];
  seg  [label="Claims · citations · quotations · speakers"];
  chk  [label="Verse-level checks\\l• cited verse exists\\l• cited verse entails the claim (NLI)\\l" +
              "• quotation matches the text\\l• speaker is correct\\l• premise true, question in scope\\l",
        fillcolor="#FFFFFF", color="#F59E0B"];
  rep  [label="Verification report: verdict · risk · issues", fillcolor="#EEF2FF", color="#A5B4FC"];
  acc  [label="Accept", fillcolor="#DCFCE7", color="#86EFAC"];
  rep2 [label="Repair + re-verify", fillcolor="#FEF3C7", color="#FCD34D"];
  abs  [label="Abstain", fillcolor="#F1F5F9", color="#CBD5E1"];
  q -> ret -> gen -> seg -> chk -> rep;
  rep -> acc; rep -> rep2; rep -> abs;
}
"""

def _headline_numbers() -> dict:
    """Pull the headline numbers from whichever result files exist (each value may be None)."""
    out: dict = {}
    sim = C.load_json(C.RESULTS / "ugrav" / "benchmark_sim-7b.json")
    if sim:
        pol = sim.get("policies_test", {})
        det = sim.get("detection", {})
        out["ug_hall"] = pol.get("ug_rav", {}).get("hallucination_rate")
        out["never_hall"] = pol.get("never_retrieve", {}).get("hallucination_rate")
        out["always_hall"] = pol.get("always_retrieve", {}).get("hallucination_rate")
        out["ug_sel"] = pol.get("ug_rav", {}).get("selective_accuracy")
        out["ug_cov"] = pol.get("ug_rav", {}).get("coverage")
        out["ug_ret"] = pol.get("ug_rav", {}).get("retrieval_rate")
        out["auroc_fused"] = det.get("fused_with_evidence", {}).get("auroc")
        out["auroc_se"] = det.get("semantic_entropy", {}).get("auroc")
        out["closed_acc"] = sim.get("closed_book_accuracy")
        out["rag_acc"] = sim.get("rag_accuracy")
    gita = C.gita_results().get("lexical")
    if gita:
        d = gita["detectors"]
        out["vghd_f1"] = d.get("vghd", {}).get("f1")
        out["vghd_auroc"] = d.get("vghd", {}).get("auroc")
        out["vghd_fpr"] = d.get("vghd", {}).get("false_positive_rate")
        out["alb_f1"] = d.get("answer_level_nli", {}).get("f1")
        out["alb_fpr"] = d.get("answer_level_nli", {}).get("false_positive_rate")
        out["n_items"] = gita.get("vghd_all_items", {}).get("n")
    out["real"] = any(k.startswith("real:") for k in C.ugrav_results())
    return out


def _framework_card(title: str, body: str, dot: str, metrics: list[tuple[str, str, str]] | None,
                    missing: tuple[str, str], links: list[tuple[str, str]]) -> None:
    """A full-width card: description, headline numbers and links on the left, architecture on the right."""
    with st.container(border=True):
        text_col, diagram_col = st.columns([11, 9], gap="large")
        with text_col:
            st.markdown(f"#### {title}")
            st.markdown(body)
            if metrics:
                C.metric_row(metrics)
            else:
                C.missing_results_note(*missing)
            st.space("small")
            for col, (key, text) in zip(st.columns(len(links)), links):
                with col:
                    st.page_link(PAGES[key][0], label=text, icon=PAGES[key][1])
        with diagram_col:
            st.graphviz_chart(dot, width="stretch")


def home() -> None:
    C.setup_page(
        "Hallucination Detection Studio", "🔬",
        "Large language models answer fluently — and sometimes wrongly, with total confidence. This studio shows "
        "two frameworks that <b>detect</b> such hallucinations and <b>act</b> on them: <b>UG-RAV</b> measures the "
        "model's own semantic uncertainty and retrieves evidence only when it is needed, verifying the answer "
        "against that evidence before it is shown; <b>VGHD</b> exploits the fixed reference system of a closed "
        "scripture — the Bhagavad Gita — to check every citation, quotation, speaker and premise verse by verse, "
        "then repairs the answer or abstains. Everything here runs offline on your machine.",
        eyebrow="M.Tech dissertation · interactive research demo",
    )
    h = _headline_numbers()

    C.metric_row([
        ("UG-RAV hallucination rate", C.pct(h.get("ug_hall")),
         f"vs {C.pct(h.get('never_hall'))} closed-book · {C.pct(h.get('always_hall'))} vanilla RAG"),
        ("Fused-risk AUROC", C.fmt(h.get("auroc_fused")),
         f"semantic entropy alone: {C.fmt(h.get('auroc_se'))}"),
        ("VGHD F1", C.fmt(h.get("vghd_f1")), f"answer-level NLI baseline: {C.fmt(h.get('alb_f1'))}"),
        ("VGHD false-positive rate", C.pct(h.get("vghd_fpr")),
         f"answer-level NLI: {C.pct(h.get('alb_fpr'))} of correct answers flagged"),
    ])
    st.caption("UG-RAV numbers: simulation benchmark (held-out test split). VGHD numbers: synthetic Gita "
               "hallucination test set, lexical NLI (held-out test split). See the results pages for details.")
    st.space("small")

    _framework_card(
        "🧭 UG-RAV — Uncertainty-Gated Retrieval-Augmented Verification",
        "For **open-domain factual QA**. Sample the model several times, cluster the answers by meaning and measure "
        "**semantic entropy**; fuse it with self-consistency, P(True) and token probabilities into a risk **r̂₀**. "
        "Confident answers are accepted without retrieval. Uncertain ones trigger retrieval, re-sampling with "
        "evidence and NLI **verification**; the answer is shown only if the fused risk **r̂** is low and the "
        "evidence entails it — otherwise the system abstains.",
        UGRAV_DOT,
        [("Closed-book accuracy", C.pct(h.get("closed_acc")), "simulation, all questions"),
         ("UG-RAV selective acc.", C.pct(h.get("ug_sel")), f"answering {C.pct(h.get('ug_cov'), 0)} of questions"),
         ("Retrieval rate", C.pct(h.get("ug_ret")), "vanilla RAG retrieves for 100%")]
        if h.get("closed_acc") is not None else None,
        ("halludetect bench-ugrav", "UG-RAV benchmark results"),
        [("se_lab", "Semantic Entropy Lab"),
         ("ugrav_live", "Run UG-RAV live"),
         ("ugrav_results", "UG-RAV results")],
    )
    _framework_card(
        "📜 VGHD — Verse-Grounded Hallucination Detection",
        "For **retrieval-augmented QA over the Bhagavad Gita** (701 verses, Besant 1922). Because every statement "
        "can be traced to *BG chapter.verse*, the answer is split into claims and each claim is checked against the "
        "verse it cites: does the verse **exist**, does it **entail** the claim, does the **quotation** match, is "
        "the **speaker** right, and is the question's **premise** true? A policy then **accepts**, **repairs** "
        "(retargets citations, restores quotations, fixes speakers) or **abstains**.",
        VGHD_DOT,
        [("VGHD AUROC", C.fmt(h.get("vghd_auroc")), "held-out test split"),
         ("VGHD F1", C.fmt(h.get("vghd_f1")), f"answer-level NLI: {C.fmt(h.get('alb_f1'))}"),
         ("Test-set answers", f"{h.get('n_items') or '—'}", "correct + hallucinated, dev + test")]
        if h.get("vghd_f1") is not None else None,
        ("halludetect bench-gita", "VGHD benchmark results"),
        [("verifier", "Verify a Gita answer"),
         ("rag", "Gita RAG assistant"),
         ("vghd_results", "VGHD results")],
    )
    st.subheader("Where to start")
    tour = [
        ("explorer", "Hallucination Explorer",
         "The taxonomy of hallucinations, worked examples, detector families and a quick quiz."),
        ("se_lab", "Semantic Entropy Lab",
         "Type answers or sample them from the simulated LLM and watch them cluster by meaning."),
        ("ugrav_live", "UG-RAV Live",
         "Step through the gate, retrieval, verification and the final decision for any question."),
        ("ugrav_results", "UG-RAV Results",
         "Detection AUROC, risk–coverage curves, policies, retrieval budgets and popularity effects."),
        ("verifier", "Gita Verse Verifier",
         "Paste a question and an answer; see every claim checked verse by verse and repaired."),
        ("rag", "Gita RAG Assistant",
         "Ask in English, Hindi or romanised Sanskrit; inject a hallucination and see whether VGHD catches it."),
        ("vghd_results", "VGHD Results",
         "Detector comparison, per-type recall, ablations and repair outcomes."),
        ("corpus", "Corpus Browser",
         "Browse and search all 701 verses with Devanagari, IAST and English."),
    ]
    cols = st.columns(4, gap="small")
    for i, (key, title, desc) in enumerate(tour):
        with cols[i % 4], st.container(border=True, height="stretch"):
            st.page_link(PAGES[key][0], label=title, icon=PAGES[key][1])
            st.caption(desc)

    C.callout(
        "<b>An honest note on the numbers.</b> The UG-RAV <i>simulation</i> results come from a mechanistic LLM "
        "simulator (<code>halludetect.ugrav.simulator</code>) that reproduces documented phenomena — the long-tail "
        "knowledge effect, confidently memorised misconceptions, confabulation and knowledge conflicts with "
        "retrieved evidence. They <b>demonstrate the mechanisms</b>; they are not measurements of a real model. "
        "Real-model results (Qwen2.5-1.5B-Instruct on PopQA with live Wikipedia retrieval and DeBERTa-MNLI) are "
        "produced by <code>scripts/run_real_experiment.py</code> and appear on the UG-RAV Results page "
        + ("— they are available in this checkout." if h.get("real") else "once that script has been run.")
        + " The VGHD test set is synthetic: controlled perturbations of reference answers.",
    )
    C.footer()


home()
