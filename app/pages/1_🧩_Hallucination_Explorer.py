"""Hallucination Explorer — taxonomy, examples, detector families and a small quiz."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import streamlit as st  # noqa: E402

C.setup_page(
    "Hallucination Explorer", "🧩",
    "What exactly is a hallucination? Following the survey of Huang et al. (2023), outputs can be unfaithful to "
    "<b>the world</b> (factuality) or to <b>the user's input and context</b> (faithfulness). Explore the taxonomy, "
    "worked examples, the families of detectors — and test yourself.",
    eyebrow="Overview · concepts",
)

TAXONOMY_DOT = """
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.22; ranksep=0.55;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.2, margin="0.16,0.08"];
  edge [color="#94A3B8", arrowsize=0.7];
  root [label="LLM hallucination", fillcolor="#1E1B4B", fontcolor="white", color="#1E1B4B"];
  fact [label="Factuality hallucination\\n(conflicts with the world)", fillcolor="#E0E7FF", color="#A5B4FC",
        fontcolor="#1E1B4B"];
  faith [label="Faithfulness hallucination\\n(conflicts with input / context)", fillcolor="#FEF3C7",
         color="#FCD34D", fontcolor="#78350F"];
  f1 [label="Factual inconsistency", fillcolor="#EEF2FF", color="#C7D2FE", fontcolor="#1E1B4B"];
  f2 [label="Factual fabrication", fillcolor="#EEF2FF", color="#C7D2FE", fontcolor="#1E1B4B"];
  g1 [label="Instruction inconsistency", fillcolor="#FFFBEB", color="#FDE68A", fontcolor="#78350F"];
  g2 [label="Context inconsistency", fillcolor="#FFFBEB", color="#FDE68A", fontcolor="#78350F"];
  g3 [label="Logical inconsistency", fillcolor="#FFFBEB", color="#FDE68A", fontcolor="#78350F"];
  root -> fact; root -> faith;
  fact -> f1; fact -> f2;
  faith -> g1; faith -> g2; faith -> g3;
}
"""

TYPES = [
    {
        "Category": "Factuality", "Type": "Factual inconsistency",
        "What goes wrong": "States a real-world fact incorrectly.",
        "Example": "Q: Who wrote the Hindi novel Godan? → “Rabindranath Tagore.” (It was Munshi Premchand.)  "
                   "Q: Who built the Taj Mahal? → “Akbar.” (It was Shah Jahan.)",
        "Typical detectors": "Semantic entropy, self-consistency, retrieval + NLI verification (UG-RAV), FActScore",
    },
    {
        "Category": "Factuality", "Type": "Factual fabrication",
        "What goes wrong": "Invents an entity, source or reference that does not exist.",
        "Example": "“The Gita says the soul cannot be cut by weapons — see BG 2.80.” (Chapter 2 has only 72 "
                   "verses.) Or an invented journal article with plausible authors and page numbers.",
        "Typical detectors": "Reference / citation existence checks, retrieval verification, VGHD citation check",
    },
    {
        "Category": "Faithfulness", "Type": "Instruction inconsistency",
        "What goes wrong": "Ignores or deviates from the user's instruction.",
        "Example": "“Translate into Hindi: ‘The meeting is at noon.’” → “The meeting will be held at 12 PM in the "
                   "main hall.” (Answers in English instead of translating.)",
        "Typical detectors": "Instruction-following checks, task classifiers, LLM-as-judge",
    },
    {
        "Category": "Faithfulness", "Type": "Context inconsistency",
        "What goes wrong": "Contradicts the document or evidence it was given.",
        "Example": "Context: “Revenue grew 12 % year on year.” → Summary: “Revenue fell during the year.”",
        "Typical detectors": "NLI faithfulness to the source (SummaC, AlignScore, RAGAS), VGHD verse entailment",
    },
    {
        "Category": "Faithfulness", "Type": "Logical inconsistency",
        "What goes wrong": "Reasoning steps contradict each other.",
        "Example": "“Each box holds 12 pens, so 4 boxes hold 12 × 4 = 48 pens. Therefore the 4 boxes contain 52 "
                   "pens.”",
        "Typical detectors": "Step-level verification, program-aided checking, self-consistency over reasoning "
                             "paths",
    },
]

FAMILIES = [
    ("Token probability", "Predictive entropy, length-normalised PE, min-token probability (FLARE)",
     "White / grey-box (log-probs)", "No", "1 generation",
     "Confuses uncertainty about wording with uncertainty about facts"),
    ("Sampling consistency", "Self-consistency, SelfCheckGPT", "Black-box", "No", "K + 1 generations",
     "Misses consistent misconceptions (confident errors)"),
    ("Semantic entropy", "Kuhn et al. (2023); Farquhar et al. (2024)",
     "Black-box (discrete) / grey-box (weighted)", "No", "K + 1 generations + NLI clustering",
     "Confident errors; ambiguous questions raise false alarms"),
    ("Self-evaluation", "P(True), verbalised confidence", "Grey-box", "No", "+1 call",
     "Over-confidence and poor calibration"),
    ("Internal-state probes", "SAPLMA, semantic-entropy probes", "White-box (hidden states)",
     "Labelled training data", "≈ free at inference", "Needs model internals and labelled data"),
    ("Retrieval-augmented verification", "RARR, FActScore, NLI against retrieved evidence", "Black-box",
     "Retrieval corpus", "Retrieval + NLI per claim", "Retrieval misses; misleading passages"),
    ("UG-RAV (this work)", "Uncertainty gate → retrieval → NLI verification → abstain",
     "Black-box (+ optional log-probs)", "Retrieval corpus — only when uncertain",
     "K + 1 generations (+ K + 1 with evidence when gated)", "Gate thresholds must be tuned on held-out data"),
    ("VGHD (this work)", "Citation existence, verse entailment, quotation, speaker and premise checks",
     "Black-box", "Closed canonical corpus with a reference system", "Cheap: lexical / NLI per claim",
     "Domain-specific; needs a citable reference system"),
]

QUIZ = [
    {
        "title": "The Taj Mahal and Akbar",
        "text": "<b>Q:</b> Who built the Taj Mahal?<br><b>A:</b> The Taj Mahal was built by the Mughal emperor Akbar.",
        "answer": "Factual inconsistency",
        "why": "The answer refers to a real entity but states a wrong fact — the Taj Mahal was commissioned by "
               "Shah Jahan. A popular misconception like this is often produced <i>consistently</i>, so sampling "
               "alone may miss it; retrieval-based verification (UG-RAV) catches it.",
    },
    {
        "title": "A verse that does not exist",
        "text": "<b>Q:</b> Which verse of the Gita says that the soul cannot be cut by weapons?<br>"
                "<b>A:</b> This is described in BG 2.80.",
        "answer": "Factual fabrication",
        "why": "Chapter 2 of the Bhagavad Gita has only 72 verses, so the citation is invented. VGHD's "
               "citation-existence check flags it and the repair policy retargets it to BG 2.23.",
    },
    {
        "title": "Translation ignored",
        "text": "<b>User:</b> Translate into Hindi: “The meeting is at noon.”<br>"
                "<b>Model:</b> The meeting will be held at 12 PM in the main hall.",
        "answer": "Instruction inconsistency",
        "why": "The task was to translate, but the model paraphrased in English (and even added a detail). The "
               "output deviates from the instruction, not from world knowledge.",
    },
    {
        "title": "Revenue grew — or fell?",
        "text": "<b>Document:</b> “Quarterly revenue grew 12 % year on year.”<br>"
                "<b>Summary:</b> Revenue fell during the quarter.",
        "answer": "Context inconsistency",
        "why": "The summary contradicts the source it was asked to summarise. An NLI model comparing summary and "
               "source reports a contradiction.",
    },
    {
        "title": "48 or 52?",
        "text": "Each box holds 12 pens, so 4 boxes hold 12 × 4 = 48 pens. Therefore the 4 boxes contain 52 pens.",
        "answer": "Logical inconsistency",
        "why": "The final statement contradicts the model's own intermediate step (48). Step-level verification "
               "or recomputing the arithmetic exposes it.",
    },
    {
        "title": "Godan and Tagore",
        "text": "<b>Q:</b> Who wrote the Hindi novel Godan?<br><b>A:</b> It was written by Rabindranath Tagore.",
        "answer": "Factual inconsistency",
        "why": "Godan (1936) is by Munshi Premchand. The UG-RAV Live page shows how the simulated model's samples "
               "disagree (high semantic entropy), retrieval is triggered and the evidence corrects the answer.",
    },
    {
        "title": "An invented journal article",
        "text": "<b>Q:</b> Give a source showing that coffee cures diabetes.<br><b>A:</b> See Sharma et al. (2019), "
                "“Coffee and glycaemic remission”, <i>The Lancet</i> 393(4): 112–118.",
        "answer": "Factual fabrication",
        "why": "The reference looks plausible but does not exist. Checking the source against a bibliographic "
               "database (or, for the Gita, against the verse index) exposes it.",
    },
    {
        "title": "Who said it?",
        "text": "<b>Retrieved verse (BG 2.47, spoken by Krishna):</b> “Thy business is with the action only, never "
                "with its fruits …”<br><b>Answer:</b> Arjuna says that one's right is to action alone (BG 2.47).",
        "answer": "Context inconsistency",
        "why": "The answer contradicts the evidence it cites: the verse is spoken by Krishna. VGHD reports this as "
               "a <i>wrong speaker</i> error and repairs the attribution.",
    },
]
TYPE_NAMES = [t["Type"] for t in TYPES]

# ------------------------------------------------------------------ taxonomy
st.subheader("A taxonomy of hallucinations")
left, right = st.columns([3, 2], gap="large")
with left:
    st.graphviz_chart(TAXONOMY_DOT, width="stretch")
with right:
    C.callout(
        "<b>Factuality</b> hallucinations conflict with verifiable world knowledge: a wrong fact "
        "(<i>inconsistency</i>) or an invented one (<i>fabrication</i>).<br><br>"
        "<b>Faithfulness</b> hallucinations conflict with what the model was given: the user's "
        "<i>instruction</i>, the <i>context</i> or documents in the prompt, or the model's own earlier "
        "<i>reasoning</i>.<br><br>"
        "UG-RAV targets factuality errors in open-domain QA; VGHD targets both — fabricated citations and "
        "answers that misrepresent the verses they cite.",
        tone="indigo",
    )

st.subheader("Types, examples and typical detectors")
CATEGORY_CHIPS = {"Factuality": C.chip("Factuality", C.INDIGO_SOFT, C.INDIGO_DARK),
                  "Faithfulness": C.chip("Faithfulness", "#FEF3C7", "#78350F")}
st.html(C.html_table(
    [{**t, "Type": f"<b>{C.esc(t['Type'])}</b><br>{CATEGORY_CHIPS[t['Category']]}"} for t in TYPES],
    ["Type", "What goes wrong", "Example", "Typical detectors"], raw=("Type",),
    widths={"Type": "18%", "What goes wrong": "20%", "Example": "36%", "Typical detectors": "26%"},
))

# ------------------------------------------------------------------ detector families
st.subheader("Families of detectors")
st.caption("What each family needs: how much access to the model, whether a reference is required, and the "
           "cost per answer. The two frameworks in this studio (highlighted) combine families to cover each "
           "other's blind spots.")
cols = ["Family", "Representative methods", "Model access", "Reference needed", "Cost per answer", "Blind spot"]
st.html(C.html_table([{**dict(zip(cols, f)), "ours": "this work" in f[0]} for f in FAMILIES], cols,
                     strong=("Family",), highlight="ours",
                     widths={"Family": "14%", "Representative methods": "22%", "Model access": "15%",
                             "Reference needed": "15%", "Cost per answer": "15%", "Blind spot": "19%"}))

# ------------------------------------------------------------------ quiz
st.subheader("Classify the hallucination")
st.caption("Pick an example, decide which type it is, and get an explanation.")
qcol, acol = st.columns([3, 2], gap="large")
with qcol:
    idx = st.selectbox("Example", range(len(QUIZ)), format_func=lambda i: f"{i + 1}. {QUIZ[i]['title']}",
                       key="quiz_example")
    item = QUIZ[idx]
    st.html(f'<div class="hd-card">{item["text"]}</div>')
with acol:
    guess = st.radio("Which type is it?", TYPE_NAMES, index=None, key=f"quiz_guess_{idx}")
if guess is not None:
    if guess == item["answer"]:
        st.success(f"**Correct — {item['answer']}.**", icon="🎯")
    else:
        st.error(f"**Not quite.** This is a **{item['answer']}**, not a {guess.lower()}.", icon="🔁")
    C.callout(item["why"])
    detectors = next(t["Typical detectors"] for t in TYPES if t["Type"] == item["answer"])
    st.caption(f"Typical detectors for this type: {detectors}.")
else:
    st.caption("Choose an answer to reveal the explanation.")

C.footer()
