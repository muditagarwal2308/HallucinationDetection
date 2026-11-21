"""UG-RAV Live — run Uncertainty-Gated Retrieval-Augmented Verification step by step."""

from __future__ import annotations

import random
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from halludetect.text import answer_to_statement  # noqa: E402
from halludetect.ugrav import UGRAV  # noqa: E402
from halludetect.ugrav.pipeline import DEFAULT_THRESHOLDS  # noqa: E402
from halludetect.uncertainty import max_entropy  # noqa: E402

ctx = C.setup_page(
    "UG-RAV Live", "🧭",
    "Watch the pipeline decide, one question at a time: sample the model closed-book, measure its semantic "
    "uncertainty, <b>gate</b> retrieval on the fused risk r̂₀, verify the evidence-grounded answer with NLI and "
    "finally <b>accept</b>, <b>answer from evidence</b> or <b>abstain</b>.",
    eyebrow="UG-RAV · live pipeline",
)

BACKENDS = {"simulator": "Offline LLM simulator", "openai": "OpenAI-compatible server (e.g. Ollama)"}
FUSION_NAMES = {
    "closed": "logistic fusion of all closed-book signals",
    "closed_blackbox": "logistic fusion of black-box signals (no log-probs)",
    "rag": "logistic fusion of closed-book + evidence signals",
    "rag_blackbox": "logistic fusion of black-box + evidence signals",
    "heuristic": "transparent fallback (no fitted fusion model found)",
}
fusion_file = C.load_json(C.RESULTS / "ugrav" / "fusion_model.json") or {}
TUNED = {**DEFAULT_THRESHOLDS, **fusion_file.get("thresholds", {})}
world = C.get_world()
wmap = C.world_by_id()
ids = [w.id for w in world]

# ------------------------------------------------------------------ sidebar
with ctx.sidebar:
    st.markdown("**Generator**")
    backend = st.radio("Backend", list(BACKENDS), format_func=BACKENDS.get, key="ug_backend")
    if backend == "simulator":
        seed = st.number_input("Simulator seed", 0, 999, 0, key="ug_seed",
                               help="Each seed is a different simulated model with different knowledge.")
        profile = st.selectbox("Simulator profile", ["sim-7b", "sim-8b"], key="ug_profile",
                               help="Behaviour profiles of a 7B- and an 8B-like model.")
        base_url, model = "", ""
    else:
        seed, profile = 0, "sim-7b"
        base_url = st.text_input("Base URL", "http://localhost:11434/v1", key="ug_base_url")
        model = st.text_input("Model", "qwen2.5:7b", key="ug_model")
        st.caption("Any `/v1/chat/completions` endpoint. An API key, if needed, is read from `OPENAI_API_KEY`.")
    st.markdown("**Sampling & verification**")
    k = st.slider("K samples", 3, 20, 10, key="ug_k")
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1, key="ug_temp")
    nli_name = st.selectbox("NLI back-end", list(C.NLI_CHOICES), format_func=C.NLI_CHOICES.get, key="ug_nli",
                            help=C.NLI_HELP)
    with st.expander("Decision thresholds"):
        override = st.toggle("Override the tuned thresholds", key="ug_override")
        tau_low = st.slider("τ_low — accept without retrieval below", 0.0, 1.0, float(TUNED["tau_low"]), 0.05,
                            disabled=not override, key="ug_tau_low")
        tau_high = st.slider("τ_high — answer from evidence below", 0.0, 1.0, float(TUNED["tau_high"]), 0.05,
                             disabled=not override, key="ug_tau_high")
        min_support = st.slider("Minimum evidence support", 0.0, 1.0, float(TUNED["min_support"]), 0.05,
                                disabled=not override, key="ug_min_support")
        st.caption(f"Tuned on the dev split: τ_low = {TUNED['tau_low']:.2f}, τ_high = {TUNED['tau_high']:.2f}, "
                   f"min. support = {TUNED['min_support']:.2f}.")
thresholds = {"tau_low": tau_low, "tau_high": tau_high, "min_support": min_support} if override else None


def _random_item() -> None:
    st.session_state.ug_item = random.choice(ids)


def _item_chips(item) -> str:
    chips = [C.chip(f"{item.tier} popularity", C.INDIGO_SOFT, C.INDIGO_DARK),
             C.chip(f"kind: {item.kind}", "#FEF3C7", "#78350F"),
             C.chip(f"popularity {item.popularity:,}", "#F1F5F9", "#334155"),
             C.chip(f"topic: {item.group}", "#F1F5F9", "#334155")]
    return "<div>" + "".join(chips) + "</div>"


# ------------------------------------------------------------------ question
st.subheader("Question")
qmode = st.segmented_control("Source", ["world", "custom"], key="ug_qmode", default="world", required=True,
                             format_func={"world": "Pick from the QA world (197 questions)",
                                          "custom": "Type your own"}.get)
item, question, aliases = None, "", []
if qmode == "custom":
    question = st.text_input("Your question", "Who wrote the Hindi novel Godan?", key="ug_custom_q")
    gold = st.text_input("Gold answers (optional, comma-separated)", key="ug_custom_gold",
                         help="Used only to score the final answer.")
    item = C.find_world_item(question)
    aliases = [a.strip() for a in gold.split(",") if a.strip()] or (item.aliases if item else [])
    if item is not None:
        st.caption(f"Matches world question **{item.id}** (“{item.question}”) — the simulator knows it.")
        st.html(_item_chips(item))
else:
    c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
    with c1:
        item_id = st.selectbox("World question", ids, index=ids.index("w001") if "w001" in ids else 0,
                               key="ug_item",
                               format_func=lambda i: f"{wmap[i].question}   ·   {wmap[i].tier} · {wmap[i].kind}")
    with c2:
        st.button("Random", icon=":material/casino:", on_click=_random_item, key="ug_random", width="stretch")
    item = wmap[item_id]
    question, aliases = item.question, item.aliases
    st.html(_item_chips(item))

can_run = bool(question.strip())
if backend == "simulator" and qmode == "custom" and item is None and question.strip():
    st.warning("The offline simulator only knows the 197 questions of the QA world, so it would answer "
               "“I don't know.”. Pick a world question or switch the backend to an OpenAI-compatible server.",
               icon="🧪")
    can_run = False
run = st.button("Run UG-RAV", type="primary", icon=":material/play_arrow:", key="ug_run", disabled=not can_run)


def _run_pipeline() -> None:
    try:
        if backend == "openai":
            err = C.check_server(base_url)
            if err:
                st.error(err, icon="🔌")
                return
            llm = C.make_openai_llm(model, base_url)
        else:
            llm = C.make_simulator(seed, profile)
        nli, warning = C.get_nli_safe(nli_name)
        if warning:
            st.warning(warning, icon="⚠️")
        pipe = UGRAV(llm, nli, C.get_retriever(), k=int(k), temperature=float(temperature), thresholds=thresholds)
        with st.spinner("Sampling, clustering and verifying …"):
            trace = pipe.run(question, aliases=aliases or None)
            per_passage = []
            if trace["rag"]:
                hyp_rag = answer_to_statement(question, trace["rag"]["greedy"])
                hyp_closed = answer_to_statement(question, trace["closed"]["greedy"])
                passages = trace["rag"]["passages"]
                res = nli.predict_batch([(p, hyp_rag) for p in passages] + [(p, hyp_closed) for p in passages])
                scores = {pid: s for pid, _, s in C.get_retriever().search(question, k=len(passages))}
                n = len(passages)
                for i, pid in enumerate(trace["rag"]["passage_ids"]):
                    per_passage.append({"id": pid, "text": passages[i], "bm25": scores.get(pid),
                                        "support": res[i].entailment, "contradiction": res[i].contradiction,
                                        "support_closed": res[n + i].entailment})
        st.session_state.ug_result = {
            "trace": trace, "item_id": item.id if item is not None else None, "backend": backend, "seed": int(seed),
            "profile": profile, "nli": getattr(nli, "name", nli_name), "k": int(k), "temperature": float(temperature),
            "model": model, "per_passage": per_passage, "aliases": list(aliases),
        }
    except Exception as exc:  # network errors, malformed server replies, model loading problems
        st.error(f"The pipeline failed: {type(exc).__name__}: {str(exc)[:300]}", icon="🚨")


if run:
    _run_pipeline()


# ------------------------------------------------------------------ trace rendering
def _metric(col, label_: str, value: str, sub: str = "") -> None:
    with col:
        st.html(C.metric_card(label_, value, sub))


def _yes_no(flag) -> str:
    return "—" if flag is None else ("✅ correct" if flag else "❌ wrong")


def render(res: dict) -> None:
    tr = res["trace"]
    closed, rag, th, dec = tr["closed"], tr["rag"], tr["thresholds"], tr["decision"]
    s0 = tr["steps"][0]
    r0 = float(s0["risk_r0"])
    r_hat = float(tr["steps"][1]["risk_r_hat"]) if len(tr["steps"]) > 1 else None
    k_used = len(closed["samples"])
    who = (f"simulator seed {res['seed']} ({res['profile']})" if res["backend"] == "simulator"
           else f"model {res['model']}")
    st.caption(f"Question: **{tr['question']}** · {who} · K = {k_used} · T = {res['temperature']:.1f} · "
               f"NLI: {res['nli']} · {tr['seconds']:.2f} s")

    nodes = ["Question", f"K = {k_used} closed-book samples", f"r̂₀ = {r0:.2f}"]
    if dec["retrieved"]:
        nodes += [f"≥ τ_low {th['tau_low']:.2f} → retrieve", "verify with NLI", f"r̂ = {r_hat:.2f}"]
    else:
        nodes += [f"< τ_low {th['tau_low']:.2f} → no retrieval"]
    lab = C.ACTION_STYLES.get(dec["action"], (dec["action"],))[0]
    nodes.append(f"<b>{C.esc(lab)}</b>")
    st.html(C.pipeline_strip(nodes))

    # step 1 --------------------------------------------------------------
    with st.container(border=True):
        C.step_header(1, "Closed-book sampling",
                      f"Greedy answer a₀ plus K = {k_used} samples at temperature {res['temperature']:.1f}, "
                      "without evidence; samples are clustered by meaning.", done=True)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.caption("Greedy answer a₀")
            st.html(C.answer_box(closed["greedy"]))
            st.caption("Samples, coloured by semantic cluster")
            st.html(C.cluster_chips(closed["samples"], closed["clusters"]))
            sizes = [closed["clusters"].count(c) for c in sorted(set(closed["clusters"]))]
            reps = [closed["samples"][closed["clusters"].index(c)] for c in sorted(set(closed["clusters"]))]
            C.show_fig(C.cluster_bar(sizes, reps), key="ug_closed_clusters")
        with right:
            m1, m2 = st.columns(2)
            _metric(m1, "Semantic entropy", C.fmt(closed["se"], 2), f"max ln K = {C.fmt(max_entropy(k_used), 2)}")
            _metric(m2, "Self-consistency", C.pct(closed["self_consistency"], 0), "largest cluster share")
            m3, m4 = st.columns(2)
            _metric(m3, "P(True)", C.fmt(closed.get("p_true"), 2), "model's self-evaluation")
            _metric(m4, "Lexical entropy", C.fmt(closed.get("lexical_entropy"), 2), "over strings")
            C.show_fig(C.gauge(r0, th["tau_low"], "closed-book risk r̂₀  (saffron = τ_low)"), key="ug_gauge_r0")
            st.caption(f"r̂₀ from the {FUSION_NAMES.get(s0['fusion_model'], s0['fusion_model'])}.")

    # step 2 --------------------------------------------------------------
    with st.container(border=True):
        C.step_header(2, "Uncertainty gate", "Retrieval is spent only where the model is unsure.", done=True)
        if dec["retrieved"]:
            st.warning(f"r̂₀ = {r0:.2f} ≥ τ_low = {th['tau_low']:.2f}: the closed-book answer is too risky to "
                       "accept. Retrieve evidence and verify.", icon="🔎")
        else:
            st.success(f"r̂₀ = {r0:.2f} < τ_low = {th['tau_low']:.2f}: the model is consistent and confident, so "
                       "its answer is accepted without retrieval (one retrieval call saved).", icon="⚡")

    # step 3 --------------------------------------------------------------
    with st.container(border=True):
        if not rag:
            C.step_header(3, "Retrieval and verification", "Skipped — the gate accepted the closed-book answer.")
        else:
            C.step_header(3, "Retrieval and verification",
                          "Top-3 BM25 passages; the model re-answers with the evidence; NLI checks whether the "
                          "evidence entails each answer.", done=True)
            pcols = st.columns(max(1, len(res["per_passage"])), gap="small")
            for col, p in zip(pcols, res["per_passage"]):
                with col:
                    score = f" · BM25 {p['bm25']:.1f}" if p.get("bm25") is not None else ""
                    st.html(
                        f'<div class="hd-card"><div class="hd-muted">passage {C.esc(p["id"])}{score}</div>'
                        f'<div style="margin:.35rem 0 .6rem 0">{C.esc(p["text"])}</div>'
                        f'<div class="hd-muted">entails the RAG answer: <b>{p["support"]:.2f}</b></div>'
                        f'{C.bar(p["support"], C.GREEN)}'
                        f'<div class="hd-muted">contradicts the RAG answer: <b>{p["contradiction"]:.2f}</b></div>'
                        f'{C.bar(p["contradiction"], C.RED)}'
                        f'<div class="hd-muted">entails the closed-book answer: <b>{p["support_closed"]:.2f}</b>'
                        f'</div>{C.bar(p["support_closed"], C.SLATE)}</div>'
                    )
            left, right = st.columns([3, 2], gap="large")
            with left:
                st.caption("Evidence-grounded greedy answer")
                st.html(C.answer_box(rag["greedy"]))
                st.caption("Samples with evidence, coloured by semantic cluster")
                st.html(C.cluster_chips(rag["samples"], rag["clusters"]))
                m1, m2, m3 = st.columns(3)
                _metric(m1, "SE with evidence", C.fmt(rag["se_rag"], 2), "after retrieval")
                _metric(m2, "ΔSE", f"{rag['delta_se']:+.2f}", "closed − RAG; > 0 = evidence settles it")
                _metric(m3, "Agreement", "yes" if rag["agreement"] else "no", "closed vs RAG majority")
            with right:
                m1, m2 = st.columns(2)
                _metric(m1, "Evidence support", C.fmt(rag["support_rag"], 2),
                        f"needs ≥ {th['min_support']:.2f}" if th["min_support"] > 0 else "max entailment over passages")
                _metric(m2, "Contradiction", C.fmt(rag["contradiction_rag"], 2), "max over passages")
                C.show_fig(C.gauge(r_hat, th["tau_high"], "fused risk r̂  (saffron = τ_high)"),
                           key="ug_gauge_rhat")
                st.caption(f"r̂ from the {FUSION_NAMES.get(tr['steps'][1]['fusion_model'], '')}.")

    # step 4 --------------------------------------------------------------
    with st.container(border=True):
        C.step_header(4, "Decision", "Accept, answer from evidence, or abstain.", done=True)
        shown = dec["answer"] if dec["action"] != "abstain" else "I don't know."
        st.html(C.action_banner(dec["action"], f"<b>{C.esc(shown)}</b><br>{C.esc(dec['rationale'])}"))
        if "closed_correct" in dec:
            c1, c2, c3 = st.columns(3)
            _metric(c1, "Closed-book answer", _yes_no(dec.get("closed_correct")), C.esc(closed["greedy"]))
            _metric(c2, "Evidence-grounded answer", _yes_no(dec.get("rag_correct")) if rag else "not generated",
                    C.esc(rag["greedy"]) if rag else "")
            final = "abstained" if dec["action"] == "abstain" else _yes_no(dec.get("final_correct"))
            _metric(c3, "Final output", final, "scored against the gold answers")
            gold = wmap[res["item_id"]].aliases if res["item_id"] is not None else res.get("aliases", [])
            if gold:
                st.caption(f"Gold answers: {', '.join(gold)}")
        else:
            st.caption("No gold answers were given, so correctness is not scored.")

    # under the hood ------------------------------------------------------
    if res["backend"] == "simulator" and res["item_id"] is not None:
        it = wmap[res["item_id"]]
        sim = C.make_simulator(res["seed"], res["profile"])
        with st.expander("Under the hood (simulator): what the simulated model actually believes"):
            state, dist = sim.closed_belief(it)
            panels = [("Closed-book belief", state, dist)]
            if rag:
                panels.append(("Belief after reading the passages", *sim.rag_belief(it, rag["passages"])))
            cols = st.columns(len(panels), gap="large")
            for i, (col, (title, st_name, d)) in enumerate(zip(cols, panels)):
                with col:
                    name, desc = C.SIM_STATES.get(st_name, (st_name, ""))
                    st.markdown(f"**{title}:** {name}")
                    st.caption(desc)
                    items = sorted(d.items(), key=lambda kv: -kv[1])
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in items], y=[a for a, _ in items], orientation="h",
                        marker_color=[C.GREEN if a in it.aliases else C.SLATE for a, _ in items],
                        text=[f"{v:.2f}" for _, v in items], textposition="outside", cliponaxis=False))
                    fig.update_xaxes(range=[0, 1.12], title="p(meaning)", gridcolor=C.GRID)
                    fig.update_yaxes(autorange="reversed")
                    C.show_fig(C.style_fig(fig, height=90 + 34 * len(items), legend=False), key=f"ug_belief_{i}")
            st.html(
                '<dl class="hd-kv">'
                f"<dt>Gold answer</dt><dd>{C.esc(it.answer)} (aliases: {C.esc(', '.join(it.aliases))})</dd>"
                f"<dt>Popularity</dt><dd>{it.popularity:,} ({C.esc(it.tier)} tier)</dd>"
                f"<dt>Kind</dt><dd>{C.esc(it.kind)}</dd>"
                f"<dt>Common misconception</dt><dd>{C.esc(it.misconception or '—')}</dd>"
                f"<dt>Distractors</dt><dd>{C.esc(', '.join(it.distractors) or '—')}</dd>"
                f"<dt>Gold passage in corpus</dt><dd>{'yes' if it.in_corpus else 'no'}</dd></dl>"
            )
            st.caption("Green bars are the gold answer. The probability of knowing a fact grows with popularity; "
                       "unknown facts are either confidently wrong (often a misconception) or confabulated.")


result = st.session_state.get("ug_result")
if result:
    render(result)
else:
    C.callout("Choose a question and press <b>Run UG-RAV</b>. Tip: <i>Who wrote the Hindi novel Godan?</i> shows "
              "a confused model whose samples disagree, so retrieval is triggered and the evidence corrects it; "
              "a high-popularity question such as <i>What is the capital of France?</i> is usually accepted "
              "without retrieval.", tone="indigo")

# ------------------------------------------------------------------ batch mode
st.divider()
st.subheader("Batch mode — 25 random questions")
st.caption("Runs the full pipeline on 25 random world questions with the offline simulator and the lexical NLI "
           "(a second or two), using the K, temperature, seed and thresholds from the sidebar.")
b1, b2 = st.columns([1, 4], vertical_alignment="bottom")
batch_seed = b1.number_input("Sampling seed", 0, 9999, 7, key="ug_batch_seed")
if b2.button("Run on 25 random questions", icon=":material/fact_check:", key="ug_batch"):
    lexical, _ = C.get_nli_safe("lexical")
    sim_seed = int(seed) if backend == "simulator" else 0
    sim = C.make_simulator(sim_seed, profile)
    pipe = UGRAV(sim, lexical, C.get_retriever(), k=int(k), temperature=float(temperature), thresholds=thresholds)
    rows = []
    with st.spinner("Running UG-RAV on 25 questions …"):
        for it in random.Random(int(batch_seed)).sample(world, 25):
            tr = pipe.run(it.question, it.aliases)
            d = tr["decision"]
            rows.append({
                "Question": it.question, "Tier": it.tier, "Kind": it.kind,
                "Knowledge state": C.SIM_STATES.get(sim.closed_belief(it)[0], ("?",))[0],
                "r̂₀": tr["steps"][0]["risk_r0"], "Retrieved": bool(d["retrieved"]),
                "Decision": C.ACTION_STYLES.get(d["action"], (d["action"],))[0],
                "Answer": d["answer"] if d["action"] != "abstain" else "—",
                "Correct": None if d["action"] == "abstain" else bool(d.get("final_correct")),
                "Closed-book correct": bool(d.get("closed_correct")),
            })
    st.session_state.ug_batch_result = {"rows": rows, "seed": sim_seed, "batch_seed": int(batch_seed)}

batch = st.session_state.get("ug_batch_result")
if batch:
    rows = batch["rows"]
    df = pd.DataFrame(rows)
    n = len(df)
    n_answered = sum(r["Correct"] is not None for r in rows)
    right = sum(r["Correct"] is True for r in rows)
    wrong = sum(r["Correct"] is False for r in rows)
    closed_acc = df["Closed-book correct"].mean()
    C.metric_row([
        ("Coverage", C.pct(n_answered / n, 0), f"{n_answered} of {n} answered"),
        ("Selective accuracy", C.pct(right / max(1, n_answered), 0), "accuracy of answered questions"),
        ("Hallucination rate", C.pct(wrong / n, 0), f"closed-book alone: {C.pct(1 - closed_acc, 0)}"),
        ("Retrieval rate", C.pct(df["Retrieved"].mean(), 0), "vanilla RAG: 100%"),
    ])
    left, right_col = st.columns([3, 2], gap="large")
    with left:
        show = df.copy()
        show["Correct"] = show["Correct"].map({True: "✅", False: "❌"}).fillna("— abstained")
        show["Closed-book correct"] = show["Closed-book correct"].map({True: "✅", False: "❌"})
        st.dataframe(show, hide_index=True, width="stretch", height=420, column_config={
            "r̂₀": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
            "Retrieved": st.column_config.CheckboxColumn(),
            "Question": st.column_config.TextColumn(width="large"),
        })
    with right_col:
        outcome = df["Correct"].map({True: "correct", False: "wrong"}).fillna("abstained")
        tab = pd.crosstab(df["Decision"], outcome)
        fig = go.Figure()
        for name, color in (("correct", C.GREEN), ("wrong", C.RED), ("abstained", C.SLATE)):
            if name in tab.columns:
                fig.add_bar(x=tab.index, y=tab[name], name=name, marker_color=color)
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title="questions", gridcolor=C.GRID)
        C.show_fig(C.style_fig(fig, height=330, title="Decisions and their outcomes"), key="ug_batch_chart")
        st.caption("UG-RAV abstains where neither the model nor the evidence can be trusted, trading coverage for "
                   "a much lower hallucination rate than answering everything closed-book.")

C.footer()
