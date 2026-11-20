"""Gita RAG Assistant — retrieve verses, generate a cited answer, verify it with VGHD, repair or abstain."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import streamlit as st  # noqa: E402

from halludetect.gita.pipeline import GitaQA  # noqa: E402
from halludetect.gita.simulator import MODES  # noqa: E402

ctx = C.setup_page(
    "Gita RAG Assistant", "🪔",
    "Ask about the Bhagavad Gita in <b>English</b>, <b>Hindi</b> or <b>romanised Sanskrit</b>. The assistant "
    "retrieves verses, writes a cited answer, and VGHD verifies it before you see it. With the offline simulator "
    "you can <b>inject a hallucination</b> on purpose and check whether the verifier catches it.",
    eyebrow="VGHD · retrieval-augmented QA",
)

EXAMPLES = {
    "English": ["What does Krishna say about action and its fruits?", "How can the restless mind be controlled?",
                "What happens to the soul when the body dies?"],
    "हिन्दी · Hindi": ["कर्म और उसके फल के बारे में कृष्ण क्या कहते हैं?", "मन को वश में कैसे करें?",
                       "आत्मा को शस्त्र क्यों नहीं काट सकते?"],
    "Romanised Sanskrit": ["What does Krishna teach about action and its fruits in karmaṇy evādhikāras te?",
                           "What does nainaṃ chindanti śastrāṇi say about weapons and the soul?"],
    "Tricky questions": ["Why does Arjuna say \"Time am I, laying desolate the world\"?",
                         "What does the Gita say about cryptocurrency?"],
}
MODE_LABELS = {
    "none": "No injection — faithful answer", "fabricated_citation": "Fabricated citation (verse does not exist)",
    "misattributed_citation": "Misattributed citation (wrong verse)", "misquotation": "Misquotation",
    "wrong_speaker": "Wrong speaker", "unsupported_claim": "Unsupported claim (commentary)",
    "random": "Random hallucination type",
}
BACKENDS = {"simulator": "Offline answer simulator", "openai": "OpenAI-compatible server (e.g. Ollama)"}


def _set_question(q: str) -> None:
    st.session_state.rag_q = q


if "rag_q" not in st.session_state:
    st.session_state.rag_q = EXAMPLES["English"][0]

with ctx.sidebar:
    st.markdown("**Assistant settings**")
    backend = st.radio("Generator", list(BACKENDS), format_func=BACKENDS.get, key="rag_backend")
    if backend == "simulator":
        inject = st.selectbox("Inject a hallucination", MODES, format_func=MODE_LABELS.get, key="rag_inject",
                              help="The simulator writes a faithful answer from the top verse, then applies the "
                                   "chosen perturbation.")
        base_url, model = "", ""
    else:
        inject = "none"
        base_url = st.text_input("Base URL", "http://localhost:11434/v1", key="rag_base_url")
        model = st.text_input("Model", "qwen2.5:7b", key="rag_model")
        st.caption("If VGHD finds a problem, the LLM is asked once to rewrite its answer with the verifier's "
                   "feedback (regenerate) before falling back to repair or abstention.")
    k = st.slider("Verses retrieved", 3, 8, 5, key="rag_k")
    nli_name = st.selectbox("NLI back-end", list(C.NLI_CHOICES), format_func=C.NLI_CHOICES.get, key="rag_nli",
                            help=C.NLI_HELP)
verifier, eff_nli, warning = C.get_verifier_safe(nli_name)
if warning:
    st.warning(warning, icon="⚠️")

# ------------------------------------------------------------------ examples and question
st.markdown("**Try an example**")
ex_cols = st.columns(len(EXAMPLES), gap="small")
n = 0
for col, (group, qs) in zip(ex_cols, EXAMPLES.items()):
    with col:
        st.caption(group)
        for q in qs:
            st.button(q, key=f"rag_ex_{n}", on_click=_set_question, args=(q,), width="stretch")
            n += 1

q_col, b_col = st.columns([6, 1], vertical_alignment="bottom")
with q_col:
    question = st.text_input("Your question", key="rag_q", placeholder="Ask anything about the Bhagavad Gita …")
with b_col:
    ask = st.button("Ask", type="primary", icon=":material/send:", key="rag_ask", width="stretch",
                    disabled=not question.strip())

if ask and question.strip():
    try:
        corpus = C.get_corpus()
        if backend == "openai":
            err = C.check_server(base_url)
            if err:
                raise ConnectionError(err)
            qa = GitaQA(corpus, llm=C.make_openai_llm(model, base_url), verifier=verifier, k=int(k))
        else:
            qa = GitaQA(corpus, verifier=verifier, k=int(k), simulator_mode=inject)
        with st.spinner("Retrieving, answering and verifying …"):
            res = qa.ask(question.strip())
            clean = None
            if backend == "simulator" and res["injected"] != "none":
                verses = [corpus.get(r["ref"]) for r in res["retrieved"]]
                clean, _ = qa.simulator.compose(question.strip(), [v for v in verses if v], mode="none")
        st.session_state.rag_result = {"res": res, "backend": backend, "inject": inject, "model": model,
                                       "clean": clean, "nli": eff_nli}
    except Exception as exc:
        st.error(f"Could not answer: {type(exc).__name__}: {str(exc)[:300]}", icon="🚨")

result = st.session_state.get("rag_result")
if not result:
    C.callout("Pick an example or type a question and press <b>Ask</b>. Choose a hallucination type in the "
              "sidebar to see how VGHD detects and repairs it.", tone="indigo")
    C.footer()
    st.stop()

# ------------------------------------------------------------------ result
res = result["res"]
rep, dec, final = res["report"], res["decision"], res["final_report"]
corpus = C.get_corpus()
injected = res.get("injected", "none")
gen_label = (f"simulator · injected: {C.label(injected, C.ISSUE_LABELS)}" if result["backend"] == "simulator"
             else f"LLM · {result['model']}")
verdict_label = C.VERDICT_STYLES.get(rep["verdict"], (rep["verdict"],))[0]
action_label = C.ACTION_STYLES.get(dec["action"], (dec["action"],))[0]
st.html(C.pipeline_strip([
    f"“{C.esc(res['question'][:60])}{'…' if len(res['question']) > 60 else ''}”",
    f"{len(res['retrieved'])} verses retrieved", C.esc(gen_label), f"VGHD: {C.esc(verdict_label)}",
    f"<b>{C.esc(action_label)}</b>",
]))
st.caption(f"Answered in {res['seconds']:.2f} s · NLI: {result['nli']}")

left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(border=True):
        C.step_header(1, "Generated answer", "Written from the retrieved verses, with BG citations.", done=True)
        st.html(C.answer_box(res["answer"]))
        if result["backend"] == "simulator" and injected != "none":
            clean = result.get("clean")
            if clean and clean != res["answer"]:
                with st.expander(f"What the injection changed ({C.label(injected, C.ISSUE_LABELS).lower()})"):
                    C.diff_block(clean, res["answer"], "Faithful answer before injection", "Answer with injection")

    with st.container(border=True):
        C.step_header(2, "Verse-level verification", "Citations, entailment, quotations, speakers and premise.",
                      done=True)
        st.html(C.verdict_banner(rep["verdict"], rep["score"]))
        for issue in rep["issues"]:
            _, _, _, border = C.SEVERITY_STYLES.get(issue["severity"], ("", "", "", C.SLATE))
            st.html(f'<div class="hd-issue" style="border-left-color:{border}">{C.severity_chip(issue["severity"])}'
                    f'<span class="hd-issue-title">{C.esc(C.label(issue["type"], C.ISSUE_LABELS))}</span>'
                    f'<div class="hd-issue-msg">{C.esc(issue["message"])}</div></div>')
        if result["backend"] == "simulator":
            if injected == "none":
                if rep["hallucinated"]:
                    if rep["premise"]["kind"] != "ok":
                        st.info("Nothing was injected, but the question itself is problematic — the simulator "
                                "answers every question, so VGHD flags that it accepted the premise.", icon="🛡️")
                    else:
                        st.warning("Nothing was injected, yet VGHD flagged the answer (a false positive or a "
                                   "weakness of the simulated answer).", icon="🤔")
                else:
                    st.success("No hallucination injected — and none detected.", icon="✅")
            elif injected in rep["types"]:
                st.success(f"**Caught.** VGHD detected the injected *{C.label(injected, C.ISSUE_LABELS).lower()}*.",
                           icon="🎯")
            elif rep["hallucinated"]:
                found = ", ".join(C.label(t, C.ISSUE_LABELS).lower() for t in rep["types"])
                st.info(f"**Flagged**, but classified as {found} rather than "
                        f"{C.label(injected, C.ISSUE_LABELS).lower()}.", icon="🔎")
            elif result.get("clean") == res["answer"]:
                st.info("The chosen perturbation could not be applied to this answer, so there was nothing to "
                        "detect. Try another question or hallucination type.", icon="ℹ️")
            else:
                st.error(f"**Missed.** The injected {C.label(injected, C.ISSUE_LABELS).lower()} was not detected.",
                         icon="🙈")

    with st.container(border=True):
        C.step_header(3, "Final answer", "Accept, repair (or regenerate) — or abstain.", done=True)
        st.html(C.action_banner(dec["action"], C.esc(dec["rationale"])))
        if dec["action"] in ("repair", "regenerate") and dec["answer"] != res["answer"]:
            C.diff_block(res["answer"], dec["answer"], "Generated answer", "Answer shown to the user")
        else:
            st.html(C.answer_box(dec["answer"]))
        if dec.get("edits"):
            st.markdown("Edits: " + " · ".join(f"`{e}`" for e in dec["edits"]))
        if dec["action"] in ("repair", "regenerate"):
            lab = C.VERDICT_STYLES.get(final["verdict"], (final["verdict"],))[0]
            (st.success if final["verdict"] == "faithful" else st.warning)(
                f"Re-verification of the final answer: **{lab}** (risk {C.fmt(final['score'], 2)}).", icon="🔁")
        if res.get("regenerated"):
            st.caption("The LLM rewrote its answer after receiving the verifier's feedback.")

with right:
    st.markdown("##### Retrieved verses")
    st.caption("Hybrid retrieval: English BM25, a Hindi→English concept glossary, and character n-grams over the "
               "IAST and Devanagari text, fused by reciprocal rank.")
    with st.container(height=1100, border=False):
        for rank, r in enumerate(res["retrieved"], start=1):
            v = corpus.get(r["ref"])
            if v is not None:
                st.html(C.verse_card(v, badges=C.badge(f"#{rank} · score {r['score']:.3f}", "#F1F5F9", "#334155")))

C.footer()
