"""Semantic Entropy Lab — cluster sampled answers by meaning and compare semantic with lexical entropy."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from halludetect.text import normalize_answer  # noqa: E402
from halludetect.uncertainty import (  # noqa: E402
    cluster_answers,
    lexical_entropy,
    max_entropy,
    self_consistency,
    semantic_entropy,
)

ctx = C.setup_page(
    "Semantic Entropy Lab", "🎲",
    "Ask a model the same question K times. If the answers <i>mean</i> different things, the model is probably "
    "confabulating. <b>Semantic entropy</b> (Kuhn et al., 2023; Farquhar et al., 2024) measures exactly that: it "
    "clusters the samples by bidirectional entailment and computes the entropy over meanings — not over strings.",
    eyebrow="UG-RAV · uncertainty signals",
)

PRESETS = {
    "paraphrases": {
        "label": "Paraphrases · Canberra",
        "question": "What is the capital of Australia?",
        "answers": ["Canberra", "It is Canberra.", "The capital is Canberra.", "Canberra, of course.", "Canberra"],
        "note": "Five ways of saying the same thing: one meaning, so semantic entropy is 0 — while lexical entropy "
                "counts four different strings and reports substantial (spurious) uncertainty.",
    },
    "confabulation": {
        "label": "Confabulation · rare novel (4/3/2/1)",
        "question": "Who wrote the Kannada novel Samskara?",
        "answers": ["U. R. Ananthamurthy", "It was U. R. Ananthamurthy.", "Ananthamurthy", "U. R. Ananthamurthy.",
                    "S. L. Bhyrappa", "It was S. L. Bhyrappa.", "Bhyrappa", "Kuvempu", "It is Kuvempu.",
                    "Girish Karnad"],
        "note": "Four different authors with shares 4/3/2/1: SE = −(0.4 ln 0.4 + 0.3 ln 0.3 + 0.2 ln 0.2 + "
                "0.1 ln 0.1) ≈ 1.28 nats. The model does not know the answer and invents several.",
    },
    "godan": {
        "label": "Godan · K = 5, one misconception",
        "question": "Who wrote the Hindi novel Godan?",
        "answers": ["Premchand", "Munshi Premchand", "Rabindranath Tagore", "It was Premchand.", "Premchand."],
        "note": "Four samples name Premchand (in three surface forms), one names Tagore: SE = −(0.8 ln 0.8 + "
                "0.2 ln 0.2) ≈ 0.50, whereas lexical entropy sees five distinct strings.",
    },
}
STRICTNESS = {"default": "NLI default", "strict": "Strict (both directions entail)",
              "relaxed": "Relaxed (no contradiction, not both neutral)"}


def _apply_preset() -> None:
    key = st.session_state.get("se_preset")
    if key in PRESETS:
        st.session_state.se_question = PRESETS[key]["question"]
        st.session_state.se_answers = "\n".join(PRESETS[key]["answers"])
        st.session_state.pop("se_sim_info", None)


def _mark_custom() -> None:
    st.session_state.se_preset = None
    st.session_state.pop("se_sim_info", None)


def _sample_from_simulator() -> None:
    item = C.world_by_id()[st.session_state.se_world]
    k, temp, seed = int(st.session_state.se_k), float(st.session_state.se_temp), int(st.session_state.se_seed)
    llm = C.make_simulator(seed)
    gens = llm.answer(item.question, n=k, temperature=temp)
    st.session_state.se_question = item.question
    st.session_state.se_answers = "\n".join(g.text for g in gens)
    st.session_state.se_preset = None
    st.session_state.se_sim_info = {"id": item.id, "seed": seed, "temperature": temp, "k": k}


if "se_question" not in st.session_state:
    st.session_state.se_preset = "paraphrases"
    _apply_preset()

# ------------------------------------------------------------------ sidebar settings
with ctx.sidebar:
    st.markdown("**Lab settings**")
    nli_name = st.selectbox("NLI back-end", list(C.NLI_CHOICES), format_func=C.NLI_CHOICES.get, key="se_nli",
                            help="Decides whether two answers entail each other. " + C.NLI_HELP)
    strict_mode = st.radio("Equivalence criterion", list(STRICTNESS), format_func=STRICTNESS.get, key="se_strict",
                           help="Strict: a ≡ b iff a ⊨ b and b ⊨ a. Relaxed: the rule of the reference "
                                "implementation of semantic entropy.")
nli, nli_warning = C.get_nli_safe(nli_name)
if nli_warning:
    st.warning(nli_warning, icon="⚠️")
strict = {"default": None, "strict": True, "relaxed": False}[strict_mode]

# ------------------------------------------------------------------ inputs
st.segmented_control("Presets", list(PRESETS), format_func=lambda k: PRESETS[k]["label"], key="se_preset",
                     on_change=_apply_preset)
in_col, sim_col = st.columns([3, 2], gap="large")
with in_col:
    question = st.text_input("Question", key="se_question", on_change=_mark_custom)
    raw = st.text_area("Sampled answers — one per line", key="se_answers", height=230, on_change=_mark_custom,
                       help="Paste K answers sampled from a model for the same question.")
with sim_col, st.container(border=True):
    st.markdown("**🎛️ Sample from the simulated LLM**")
    st.caption("Draw K answers for a question of the open-domain QA world from the offline LLM simulator.")
    world = C.get_world()
    wmap = C.world_by_id()
    ids = [w.id for w in world]
    st.selectbox("World question", ids, index=ids.index("w001") if "w001" in ids else 0, key="se_world",
                 format_func=lambda i: f"{wmap[i].question}  ·  {wmap[i].tier} popularity")
    s1, s2, s3 = st.columns(3)
    s1.slider("K samples", 2, 20, 10, key="se_k")
    s2.slider("Temperature", 0.1, 2.0, 1.0, step=0.1, key="se_temp",
              help="Lower temperatures sharpen the model's belief: samples agree more often.")
    s3.number_input("Seed", 0, 999, 0, key="se_seed", help="Different seeds are different simulated models.")
    st.button("Sample answers", icon=":material/casino:", key="se_sample", on_click=_sample_from_simulator,
              type="primary", width="stretch")

answers = [line.strip() for line in (raw or "").splitlines() if line.strip()][:50]
preset = st.session_state.get("se_preset")
if preset in PRESETS:
    C.callout(f"<b>Preset:</b> {C.esc(PRESETS[preset]['note'])}", tone="indigo")
info = st.session_state.get("se_sim_info")
if info:
    item = C.world_by_id().get(info["id"])
    if item is not None:
        state, dist = C.make_simulator(info["seed"]).closed_belief(item)
        st.caption(f"Sampled {info['k']} answers from simulator seed {info['seed']} at temperature "
                   f"{info['temperature']:.1f}. Gold answer: **{item.answer}**.")
        with st.expander("Under the hood: the simulator's belief for this question"):
            name, desc = C.SIM_STATES.get(state, (state, ""))
            st.markdown(f"Knowledge state: **{name}** — {desc}")
            belief = pd.DataFrame({"answer": list(dist), "probability": list(dist.values())})
            belief["gold"] = belief["answer"].isin(item.aliases)
            fig = go.Figure(go.Bar(x=belief["probability"], y=belief["answer"], orientation="h",
                                   marker_color=[C.GREEN if g else C.SLATE for g in belief["gold"]],
                                   text=[f"{p:.2f}" for p in belief["probability"]], textposition="outside",
                                   cliponaxis=False))
            fig.update_xaxes(range=[0, 1.1], title="belief p(meaning)", gridcolor=C.GRID)
            fig.update_yaxes(autorange="reversed")
            C.show_fig(C.style_fig(fig, height=80 + 34 * len(belief), legend=False), key="se_belief")
            st.caption("Green = the gold answer. Sampling at temperature T draws meanings from this distribution "
                       "(sharpened or flattened by 1/T) and renders each in a random surface form.")

if len(answers) < 2:
    st.info("Enter at least two answers (one per line) to measure their uncertainty.", icon="✍️")
    C.footer()
    st.stop()

# ------------------------------------------------------------------ computation
clusters = cluster_answers(answers, nli, question, strict=strict)
k = len(answers)
se = semantic_entropy(clusters)
le = lexical_entropy(answers)
mx = max_entropy(k)
sc = self_consistency(clusters)
reps = clusters.representatives or [answers[c[0]] for c in clusters.clusters]

st.subheader("Result")
C.metric_row([
    ("Samples K", str(k), f"{len(set(map(normalize_answer, answers)))} distinct strings"),
    ("Meaning clusters", str(clusters.n_clusters), f"{clusters.nli_calls} NLI equivalence tests"),
    ("Semantic entropy", C.fmt(se, 2), f"{C.pct(se / mx if mx else 0, 0)} of the maximum"),
    ("Lexical entropy", C.fmt(le, 2), "entropy over distinct strings"),
    ("Maximum ln K", C.fmt(mx, 2), "all samples mean different things"),
    ("Self-consistency", C.pct(sc, 0), "share of the largest cluster"),
])
st.space("small")

if se < 1e-9 and le > 1e-9:
    verdict = (f"All {k} samples share <b>one meaning</b>: semantic entropy is 0. Lexical entropy ({C.fmt(le, 2)}) "
               "is non-zero only because the same answer is phrased differently — it overstates uncertainty.")
elif se < 1e-9:
    verdict = f"All {k} samples are identical: there is no uncertainty at all."
elif se / mx > 0.6:
    verdict = (f"The samples disagree in meaning ({clusters.n_clusters} clusters, SE = {C.fmt(se, 2)}, "
               f"{C.pct(se / mx, 0)} of the maximum): a hallmark of <b>confabulation</b> — the model is guessing.")
else:
    verdict = (f"Most samples agree (largest cluster {C.pct(sc, 0)}), but {clusters.n_clusters - 1} alternative "
               f"meaning(s) appear: SE = {C.fmt(se, 2)} versus lexical entropy {C.fmt(le, 2)}.")
C.callout(verdict)

st.markdown("**Samples coloured by meaning cluster**")
st.html(C.cluster_chips(answers, clusters.assignments))

c1, c2 = st.columns(2, gap="large")
with c1:
    C.show_fig(C.cluster_bar(clusters.sizes, reps, title="Distribution over meanings → semantic entropy"),
               key="se_cluster_bar")
with c2:
    counts = Counter(normalize_answer(a) for a in answers)
    first_form = {}
    for a in answers:
        first_form.setdefault(normalize_answer(a), a)
    cluster_of_form = {normalize_answer(a): c for a, c in zip(answers, clusters.assignments)}
    forms = list(counts)
    fig = go.Figure(go.Bar(
        x=[first_form[f] for f in forms], y=[counts[f] / k for f in forms],
        marker_color=[C.CLUSTER_SOLID[cluster_of_form[f] % len(C.CLUSTER_SOLID)] for f in forms],
        text=[f"{counts[f] / k:.2f}" for f in forms], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>p = %{y:.2f}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 1.12], title="p(string)", gridcolor=C.GRID)
    fig.update_xaxes(title="distinct normalised strings (coloured by cluster)", tickangle=-30)
    fig.update_layout(bargap=0.55 if len(forms) < 3 else 0.3)
    C.show_fig(C.style_fig(fig, height=300, legend=False, title="Distribution over strings → lexical entropy"),
               key="se_string_bar")

table = pd.DataFrame([
    {"Cluster": f"C{i + 1}", "Representative": reps[i] if i < len(reps) else answers[members[0]],
     "Members": " · ".join(answers[j] for j in members), "Count": len(members), "p(c | x)": len(members) / k}
    for i, members in enumerate(clusters.clusters)
])
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "p(c | x)": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
    "Members": st.column_config.TextColumn(width="large"),
})

# ------------------------------------------------------------------ explanations
st.subheader("How it works")
t1, t2, t3 = st.tabs(["Formulas", "Why lexical entropy overstates uncertainty", "Clustering by entailment"])
with t1:
    st.latex(r"\mathrm{SE}(x) \;=\; -\sum_{c} p(c \mid x)\,\ln p(c \mid x), \qquad p(c \mid x) = \frac{|c|}{K}")
    st.latex(r"\mathrm{LE}(x) \;=\; -\sum_{s} \hat p(s \mid x)\,\ln \hat p(s \mid x)"
             r"\qquad (s:\ \text{distinct normalised strings})")
    st.latex(r"0 \;\le\; \mathrm{SE}(x) \;\le\; \mathrm{LE}(x) \;\le\; \ln K")
    st.markdown(
        "The *discrete* (black-box) estimator above only needs the sampled texts. When token log-probabilities "
        "are available, the *weighted* estimator replaces |c|/K with the normalised probability mass of the "
        "samples in each cluster. UG-RAV uses both, together with self-consistency, P(True) and predictive "
        "entropy, and fuses them into the closed-book risk r̂₀."
    )
with t2:
    st.markdown(
        f"Lexical entropy treats *“Canberra”* and *“It is Canberra.”* as two different outcomes. Every harmless "
        f"paraphrase therefore adds uncertainty, so a model that always gives the right answer in varied words "
        f"looks unsure. Because every meaning cluster is a union of distinct strings, merging strings can only "
        f"lower the entropy: **SE ≤ LE**. In this example the gap is **{C.fmt(le - se, 2)} nats**. On the "
        f"simulation benchmark, semantic entropy detects wrong answers better than lexical entropy (see "
        f"*UG-RAV Results*)."
    )
with t3:
    st.markdown(
        "Answers are clustered greedily (Algorithm 1 of Kuhn et al., 2023): each answer is compared with the "
        "first member of every existing cluster and joins the first cluster it is **equivalent** to; otherwise "
        "it opens a new cluster. Two answers *a* and *b* are equivalent when, with the question *q* as context, "
        "each entails the other:"
    )
    st.latex(r"a \equiv b \iff \mathrm{NLI}(q \oplus a \Rightarrow q \oplus b) = \text{entail} \;\wedge\; "
             r"\mathrm{NLI}(q \oplus b \Rightarrow q \oplus a) = \text{entail}")
    st.markdown(
        "Answers that are identical after normalisation are merged without an NLI call. The offline **lexical "
        "NLI** compares content words (with stemming and a synonym lexicon); **DeBERTa-large-MNLI** is the model "
        "used in the semantic-entropy papers and can be selected in the sidebar."
    )

C.footer()
